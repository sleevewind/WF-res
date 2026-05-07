# WFRes Code Submission

This directory contains the code used for the experiments reported in the paper.
It is organized into three parts:

- `main.py` and `models/wfres.py` for ImageNet classification
- `object_detection/` for COCO object detection / instance segmentation
- `semantic_segmentation/` for ADE20K semantic segmentation

Only the files needed for the submitted experiments are kept here.

## Environment

The experiments were run in a Python 3.8 environment.

Known package versions:

- `python==3.8.20`
- `torch==1.8.0+cu111`
- `torchvision==0.9.0+cu111`
- `numpy==1.23.5`
- `timm==0.3.2`
- `mmcv-full==1.3.17`
- `mmdet==2.11.0`
- `mmsegmentation==0.30.0`
- `mmpycocotools==12.0.3`
- `opencv-python-headless`

If any of these commands fail, the missing package should be installed in the server environment before running the corresponding experiment.

## 1. Classification

The classification code is at the root of this directory.

Main model file:

- `models/wfres.py`

Main training entry:

- `main.py`

Example training command:

```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model wfres \
  --drop_path 0.1 \
  --batch_size 128 \
  --lr 4e-3 \
  --update_freq 4 \
  --model_ema true \
  --model_ema_eval true \
  --data_path ../imagenet \
  --output_dir output/wfres_lr4e-3
```

The classification checkpoint used for downstream transfer is:

- `output/wfres_lr4e-3/ckpt_model_ema.pth`

## 2. Object Detection

The detection code is under:

- `object_detection/`

The submitted detection config is:

- `object_detection/configs/wfres/cascade_mask_rcnn_wfres_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py`

Example training command:

```bash
cd object_detection
bash tools/dist_train.sh \
  configs/wfres/cascade_mask_rcnn_wfres_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py \
  8 \
  --cfg-options model.pretrained=../output/wfres_lr4e-3/ckpt_model_ema.pth
```

Example evaluation command:

```bash
cd object_detection
bash tools/dist_test.sh \
  configs/wfres/cascade_mask_rcnn_wfres_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py \
  /path/to/checkpoint.pth \
  8 \
  --eval bbox segm
```

### Detection FLOPs / Params

Example command:

```bash
cd object_detection
PYTHONPATH=$(pwd):$PYTHONPATH \
python tools/analysis_tools/get_flops.py \
  configs/wfres/cascade_mask_rcnn_wfres_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py \
  --shape 1280 800
```

### Detection FPS

Example command:

```bash
cd object_detection
PYTHONPATH=$(pwd):$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs/wfres/cascade_mask_rcnn_wfres_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py \
  /path/to/checkpoint.pth \
  --log-interval 50
```

If only architecture speed is needed, the checkpoint argument can be omitted.

## 3. Semantic Segmentation

The segmentation code is under:

- `semantic_segmentation/`

The submitted segmentation config is:

- `semantic_segmentation/configs/wfres/upernet_wfres_tiny_512_160k_ade20k_ms.py`

Example training command:

```bash
cd semantic_segmentation
bash tools/dist_train.sh \
  configs/wfres/upernet_wfres_tiny_512_160k_ade20k_ms.py \
  8 \
  --seed 0 \
  --deterministic \
  --options model.pretrained=../output/wfres_lr4e-3/ckpt_model_ema.pth
```

Example evaluation command:

```bash
cd semantic_segmentation
bash tools/dist_test.sh \
  configs/wfres/upernet_wfres_tiny_512_160k_ade20k_ms.py \
  /path/to/checkpoint.pth \
  8 \
  --eval mIoU
```

Multi-scale + flip evaluation:

```bash
cd semantic_segmentation
bash tools/dist_test.sh \
  configs/wfres/upernet_wfres_tiny_512_160k_ade20k_ms.py \
  /path/to/checkpoint.pth \
  8 \
  --eval mIoU \
  --aug-test
```

### Segmentation FLOPs / Params

Example command:

```bash
cd semantic_segmentation
python tools/get_flops.py \
  configs/wfres/upernet_wfres_tiny_512_160k_ade20k_ms.py \
  --shape 2048 512
```
