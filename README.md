# mmdet_module

```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

```
pip install "mmsegmentation>=1.0.0"
mim install mmdet
```

```
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

