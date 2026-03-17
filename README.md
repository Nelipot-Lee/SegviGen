# SegviGen: Repurposing 3D Generative Model for Part Segmentation

![teaser](assets/teaser.png)

## 🏠 [Project Page](https://fenghora.github.io/SegviGen-Page/) | [Paper](https://fenghora.github.io/SegviGen-Page/) | [Online Demo](https://huggingface.co/spaces/fenghora/SegviGen)

***SegviGen*** is a framework for 3D part segmentation that leverages the rich 3D structural and textural knowledge encoded in large-scale 3D generative models. 
It learns to predict part-indicative colors while reconstructing geometry, and unifies three settings in one architecture: **interactive part segmentation**, **full segmentation**, and **2D segmentation map–guided full segmentation** with arbitrary granularity.


## 🌟 Features
- **Repurposed 3D Generative Priors for Data Efficiency**: By reusing the rich structural and textural knowledge encoded in large-scale native 3D generative models, ***SegviGen*** learns 3D part segmentation with minimal task-specific supervision, requiring only **0.32%** training data.
- **Unified and Flexible Segmentation Settings**: Supports **interactive part segmentation**, **full segmentation**, and **2D segmentation map–guided full segmentation** with arbitrary part granularity under a single architecture.
- **State-of-the-Art Accuracy**: Consistently surpasses P3-SAM, delivering a **40%** gain in IoU@1 for single-click interaction on PartObjaverse-Tiny and PartNeXT, and a **15%** improvement in overall IoU for unguided full segmentation averaged across datasets.


## 🔨Installation

### Prerequisites
- **System**: Linux
- **GPU**: A NVIDIA GPU with at least 24GB of memory is necessary
- **Python**: 3.10

### Installation Steps
1. Create the environment of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    ```

2. Install the rest of requirements
    ```sh
    pip install mathutils
    pip install transformers==4.57.6 # https://github.com/microsoft/TRELLIS.2/issues/101
    pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
    sudo apt-get install -y libsm6 libxrender1 libxext6
    pip install --upgrade Pillow
    ```

3. If want to train
    ```sh
    pip install pytorch_lightning
    ```

### Pretrained Weights

The checkpoints of **Interactive part-segmentation**, **Full segmentation** and **Full segmentation with 2D guidance** are available on [Hugging Face](https://huggingface.co/fenghora/SegviGen).

## 📒Usage

- **Interactive part-segmentation**
    ```sh
    python inference_interactive.py \
        --ckpt_path path/to/interactive_seg.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --transforms ./data_toolkit/transforms.json \
        --img ./data_toolkit/assets/img.png \
        --export_glb ./data_toolkit/assets/output.glb \
        --input_vxz_points 388 448 392
    ```

- **Full segmentation**
    ```sh
    python inference_full.py \
        --ckpt_path path/to/full_seg.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --transforms ./data_toolkit/transforms.json \
        --img ./data_toolkit/assets/img.png \
        --export_glb ./data_toolkit/assets/output.glb
    ```

- **Full segmentation with 2D guidance**
    ```sh
    python inference_full.py \
        --ckpt_path path/to/full_seg_w_2d_map.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --img ./data_toolkit/assets/full_seg_w_2d_map/2d_map.png \
        --export_glb ./data_toolkit/assets/output.glb \
        --two_d_map
    ```

## 🔥Training

### Data preparation

 - `data_toolkit/example_interactive_seg.py`: **Interactive part-segmentation**
 - `data_toolkit/example_full_seg.py`: **Full segmentation**
 - `data_toolkit/example_full_seg_w_2d_map.py`: **Full segmentation with 2D guidance**

### Running training

- **Interactive part-segmentation**
    ```sh
    python train_interactive.py \
        --dataset_path ./data_toolkit/assets/interactive_seg/dataset.json \
        --ckpts_path path/to/ckpts_interactive_seg
    ```

- **Full segmentation**

    ```sh
    python train_full.py \
        --dataset_path ./data_toolkit/assets/full_seg/dataset.json \
        --ckpts_path path/to/ckpts_full_seg
    ```

- **Full segmentation with 2D guidance**
    ```sh
    python train_full.py \
        --dataset_path ./data_toolkit/assets/full_seg_w_2d_map/dataset.json \
        --ckpts_path path/to/ckpts_full_seg_w_2d_map
    ```
