import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import random
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import trellis2.modules.sparse as sp

from trellis2 import models
from types import MethodType
from torch.nn import functional as F
from pytorch_lightning import Trainer
from trellis2.modules.utils import manual_cast
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

def flow_forward(self, x, t, cond, concat_cond, point_embeds, coords_len_list):
    # x.feats: [N, 32]
    x = sp.sparse_cat([x, concat_cond], dim=-1)
    if isinstance(cond, list):
        cond = sp.VarLenTensor.from_tensor_list(cond)
    # x.feats: [N, 64]
    h = self.input_layer(x)
    # h.feats: [N, 1536]
    h = manual_cast(h, self.dtype)
    t_emb = self.t_embedder(t)
    t_emb = self.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, self.dtype)
    cond = manual_cast(cond, self.dtype)
    point_embeds = manual_cast(point_embeds, self.dtype)

    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        h_feats_list.append(point_embeds.feats[i*10:(i+1)*10])
        h_coords_list.append(point_embeds.coords[i*10:(i+1)*10])
        begin = end + 10
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))

    for block in self.blocks:
        h = block(h, t_emb, cond)

    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        begin = end
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))

    h = manual_cast(h, x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    # h.feats: [N, 1536]
    h = self.out_layer(h)
    # h.feats: [N, 32]
    return h

class Gen3DSeg(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)
        
    def get_positional_encoding(self, input_points):
        point_feats_embed = torch.zeros((10, 1536), dtype=torch.float32).to(input_points['point_slats'].feats.device)
        labels = input_points['point_labels'].squeeze(-1)
        point_feats_embed[labels == 1] = self.seg_embeddings.weight
        return sp.SparseTensor(point_feats_embed, input_points['point_slats'].coords)

    def forward(self, x_t, tex_slats, shape_slats, t, cond, input_points, coords_len_list):
        input_tex_feats_list = []
        input_tex_coords_list = []
        shape_feats_list = []
        shape_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list.append(x_t.feats[begin:end])
            input_tex_feats_list.append(tex_slats.feats[begin:end])
            input_tex_coords_list.append(x_t.coords[begin:end])
            input_tex_coords_list.append(tex_slats.coords[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            begin = end
        x_t = sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list))
        shape_slats = sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list))

        point_embeds = self.get_positional_encoding(input_points)
        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats, point_embeds, coords_len_list)
        
        output_tex_feats_list = []
        output_tex_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            output_tex_feats_list.append(output_tex_slats.feats[begin:end])
            output_tex_coords_list.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        output_tex_slat = sp.SparseTensor(torch.cat(output_tex_feats_list), torch.cat(output_tex_coords_list))
        return output_tex_slat

class Gen3DSegDataset(Dataset):
    def __init__(self, dataset_path, indices, split="train", repeat=1):
        super().__init__()
        self.repeat = repeat
        self.split = split
        self.indices = indices
        with open(dataset_path, "r") as f:
            all_samples = json.load(f)
        if self.indices == -1:
            self.indices = [0, len(all_samples)]
        self.all_samples = self.split_data(all_samples, split)

    def split_data(self, all_samples, split):
        repeat = self.repeat if split == "train" else 1
        all_samples = all_samples[self.indices[0] : self.indices[1]]
        all_samples = all_samples * repeat
        return all_samples

    def __len__(self):
        return len(self.all_samples)
    
    def load_instance(self, index):
        shape_slat = torch.load(self.all_samples[index]["shape_slat"])
        shape_slat = sp.SparseTensor(shape_slat["feats"], shape_slat["coords"])
        input_tex_slat = torch.load(self.all_samples[index]["input_tex_slat"])
        input_tex_slat = sp.SparseTensor(input_tex_slat["feats"], input_tex_slat["coords"])
        output_tex_slat_gt = torch.load(self.all_samples[index]["output_tex_slat_gt"])
        output_tex_slat_gt = sp.SparseTensor(output_tex_slat_gt["feats"], output_tex_slat_gt["coords"])
        cond_dict = torch.load(self.all_samples[index]["cond"])
        max_point_num = self.all_samples[index]["max_point_num"]
        point_num = random.randint(1, max_point_num)
        input_points = torch.load(self.all_samples[index]["input_points"].format(point_num=point_num))
        return {"shape_slat": shape_slat, "input_tex_slat": input_tex_slat, "output_tex_slat_gt": output_tex_slat_gt, "cond_dict": cond_dict, "input_points": input_points}
    
    def __getitem__(self, index):
        try:
            return self.load_instance(index)
        except Exception as e:
            print(f"Error in {self.all_samples[index]}: {e}")
            return self.__getitem__((index + 1) % self.__len__())

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, dataset_path, indices, repeat, shuffle, seed):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.indices = indices
        self.repeat = repeat
        self.shuffle = shuffle
        self.seed = seed

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = Gen3DSegDataset(self.dataset_path, self.indices, "train", self.repeat)

    def collate_fn(self, batch):
        shape_slats = sp.sparse_cat([sample["shape_slat"] for sample in batch])
        input_tex_slats = sp.sparse_cat([sample["input_tex_slat"] for sample in batch])
        output_tex_slat_gts = sp.sparse_cat([sample["output_tex_slat_gt"] for sample in batch])
        cond_dicts = [sample["cond_dict"] for sample in batch]
        point_slats = sp.sparse_cat([sp.SparseTensor(sample["input_points"]["point_feats"], sample["input_points"]["point_feats"]) for sample in batch])
        point_labels = torch.cat([sample["input_points"]["point_labels"] for sample in batch])
        input_points = {'point_slats': point_slats, 'point_labels': point_labels}
        coords_len_list = [sample["shape_slat"].coords.shape[0] for sample in batch]
        return {"shape_slats": shape_slats, "input_tex_slats": input_tex_slats, "output_tex_slat_gts": output_tex_slat_gts, "cond_dicts": cond_dicts, "input_points": input_points, "coords_len_list": coords_len_list}

    def train_dataloader(self):
        distributed_sampler = None
        if hasattr(self.trainer, "world_size") and self.trainer.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            distributed_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=self.shuffle,
                seed=self.seed
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=distributed_sampler,
            shuffle=False,
        )

class System(pl.LightningModule):
    def __init__(self, gen3dseg, pipeline_args, sigma_min, p_uncond, print_every):
        super().__init__()
        self.gen3dseg = gen3dseg
        self.sigma_min = sigma_min
        self.p_uncond = p_uncond
        self.print_every = print_every
        self.shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None]
        self.shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None]
        self.tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None]
        self.tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None]
        for param in self.gen3dseg.parameters():
            param.requires_grad = True
        self.gen3dseg.train()

    def forward(self, shape_slats, input_tex_slats, output_tex_slat_gts, cond_dicts, input_points, coords_len_list):
        batch_size = len(coords_len_list)
        device = shape_slats.feats.device
        shape_slats = ((shape_slats - self.shape_mean.to(device)) / self.shape_std.to(device))
        input_tex_slats = ((input_tex_slats - self.tex_mean.to(device)) / self.tex_std.to(device))

        x_0 = (output_tex_slat_gts - self.tex_mean.to(device)) / self.tex_std.to(device)
        t = torch.sigmoid(torch.randn(batch_size) * 1.0 + 1.0).to(device)
        t_x = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        noise = sp.SparseTensor(torch.randn_like(x_0.feats), x_0.coords).to(device)
        x_t = (1 - t_x) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t_x) * noise

        mask = list(np.random.rand(batch_size) < self.p_uncond)
        cond_list = []
        for i in range(batch_size):
            if mask[i]:
                cond_list.append(cond_dicts[i]['neg_cond'])
            else:
                cond_list.append(cond_dicts[i]['cond'])
        cond = torch.cat(cond_list, dim=0)

        pred = self.gen3dseg(x_t, input_tex_slats, shape_slats, t*1000, cond, input_points, coords_len_list)
        
        target = (1 - self.sigma_min) * noise - x_0
        loss = F.mse_loss(pred.feats, target.feats)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.gen3dseg.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler = {"scheduler": torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=9999999), "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self(batch["shape_slats"], batch["input_tex_slats"], batch["output_tex_slat_gts"], batch["cond_dicts"], batch["input_points"], batch["coords_len_list"])
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()

        if (self.global_step + 1) % self.print_every == 0:
            self.print(f"[step {self.global_step+1}] train_loss = {loss.item():.6f}")
        return loss

def train(dataset_path, ckpts_path):
    pl.seed_everything(42, workers=True)
    data_module = DataModule(1, 16, dataset_path, -1, 1, True, 42)

    with open("microsoft/TRELLIS.2-4B/pipeline.json", "r") as f:
        pipeline_config = json.load(f)
    pipeline_args = pipeline_config['args']
    tex_slat_flow_model = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    tex_slat_flow_model.forward = MethodType(flow_forward, tex_slat_flow_model)

    gen3dseg = Gen3DSeg(tex_slat_flow_model)
    sigma_min = pipeline_args['tex_slat_sampler']['args']['sigma_min']
    system = System(gen3dseg, pipeline_args, sigma_min, p_uncond=0.1, print_every=10)
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpts_path,
        filename="step_{step}",
        every_n_train_steps=500,
        save_top_k=-1
    )
    trainer = Trainer(
        callbacks=[ckpt_callback],
        accelerator="gpu",
        devices=-1,
        max_epochs=1,
        gradient_clip_val=1.0
    )
    trainer.fit(system, datamodule=data_module)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset.json.",
    )
    parser.add_argument(
        "--ckpts_path",
        type=str,
        required=True,
        help="Directory to save checkpoints.",
    )
    args = parser.parse_args()
    train(args.dataset_path, args.ckpts_path)
