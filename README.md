## Introduction

**MANet** is an open-source  semantic segmentation method based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models).


## Install

Open the folder **GeoLab** using **Linux Terminal** and create python environment:
```
conda create -n OMGD python=3.8
conda activate OMGD

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r MANet/requirements.txt
```

## Training

```
python MANet/train_model.py -c MANet/config/vaihingen/manet.py
```
Use different **config** to train different models.

## Inference on huge remote sensing image
```
python MANet/inference/inference_huge_image.py \
-i data/vaihingen/test_images \
-c MANet/config/vaihingen/manet.py \
-o fig_results/vaihingen/manet_huge \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```

## Acknowledgement

We wish **MANet** could serve the growing research of remote sensing by providing an effective method and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **MANet**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
