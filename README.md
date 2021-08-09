## DMLNet for open world semantic segmentation

This code is for ICCV2021 paper: "Deep Metric Learning for OpenWorld Semantic Segmentation".

The open-set semantic segmentation module is in *./anomaly* and the incremental few-shot learning module is in *./DeepLabV3Plus-Pytorch*.

### Open-set semantic segmentation module
We provide the procedure to reproduce the DML results of Table 1 in the paper.
#### Dataset
Download official StreetHazards training [dataset](https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar) and test [dataset](https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar), then extract it as following:
```
YourPath/Open_world
            /DeepLabV3Plus-Pytorch
            /anomaly
            /data
                /streethazards
                    /train
                    /test
```
#### Pretrained model
Download pretrained model [checkpoints](https://drive.google.com/drive/folders/1hJh6x42ggQU55VTDkZybiKWpBnK-FmXX?usp=sharing), and put them under *./anomaly/ckpt/*.
#### Evaluation
All codes of open-set semantic segmentation are in *./anomaly*, so first:
```bash
cd anomaly
```
Then set the dataset route:
```bash
vim config/test_ood_street.yaml
change the root_dataset, list_train and list_val as:
  root_dataset: "YourPath/Open_world/data/streethazards/test"
  list_train: "YourPath/Open_world/data/streethazards/train/train.odgt"
  list_val: "YourPath/Open_world/data/streethazards/test/test.odgt"
```
Now evaluate:
```bash
python eval_ood_traditional.py --ood dissum
```

### Incremental few-shot learning module

We provide the procedure to reproduce '16+1 setting 5 shot' results of Table 4 in the paper.
#### Dataset

Download Cityscapes dataset and extract it as following:

```
YourPath/Open_world
            /DeepLabV3Plus-Pytorch
            /anomaly
            /data
                /streethazards
                    /train
                    /test
                /cityscapes
                    /gtFine
                    /leftImg8bit
```

#### Pretrained model
Download the pretrained models from
[checkpoints](https://drive.google.com/drive/folders/1pKTj0QAiK493Pv2eqXfIYMCggsgZ-8LX?usp=sharing).
Put 4 *pth* files into the *./DeepLabV3Plus-Pytorch/ckpt/*, and *prototype_car_5_shot.json* into *./DeepLabV3Plus-Pytorch*.

#### 16+1 setting 5 shot evaluation
All codes of incremental few-shot learning module are in *./DeepLabV3Plus-Pytorch*. So first:
```bash
cd DeepLabV3Plus-Pytorch
```

FT
```bash
python test_self_distillation.py --model deeplabv3plus_embedding_self_distillation_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 5 --output_stride 16 --data_root ../data/cityscapes --total_itrs 10 --val_interval 10 --novel_cls 1 --ckpt ./ckpt/best_deeplabv3plus_embedding_self_distillation_resnet101_cityscapes_161_FT.pth --test_only
```

PLM<sub>all</sub>
```bash
python test_self_distillation.py --model deeplabv3plus_embedding_self_distillation_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 5 --output_stride 16 --data_root ../data/cityscapes --total_itrs 10 --val_interval 10 --novel_cls 1 --ckpt ./ckpt/best_deeplabv3plus_embedding_self_distillation_resnet101_cityscapes_161_5_shot_PLM.pth --test_only
```

PLM<sub>latest</sub>
```bash
vim test_self_distillation.py
comment out line 292, 295-297
uncomment line 293
```
```bash
python test_self_distillation.py --model deeplabv3plus_embedding_self_distillation_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 5 --output_stride 16 --data_root ../data/cityscapes --total_itrs 10 --val_interval 10 --novel_cls 1 --ckpt ./ckpt/best_deeplabv3plus_embedding_self_distillation_resnet101_cityscapes_161_5_shot_PLM.pth --test_only
```

NPM
```bash
python test_embedding.py --model deeplabv3plus_embedding_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 16 --output_stride 16 --data_root ../data/cityscapes --test_only --ckpt ./ckpt/best_deeplabv3plus_embedding_resnet101_cityscapes_131415.pth
```

All 17
```bash
vim test_embedding.py
change line 661 opts.num_classes to 17
comment out line 428-451
```
```bash
python test_embedding.py --model deeplabv3plus_embedding_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 16 --output_stride 16 --data_root ../data/cityscapes --test_only --ckpt ./ckpt/best_deeplabv3plus_embedding_resnet101_cityscapes_1415.pth
```

First 16
```bash
vim datasets/cityscapes.py
change line 71 unknown target to [13,14,15]
vim test_embedding.py
change line 661 opts.num_classes to 16
comment out line 428-451
```

```bash
python test_embedding.py --model deeplabv3plus_embedding_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 16 --output_stride 16 --data_root ../data/cityscapes --test_only --ckpt ./ckpt/best_deeplabv3plus_embedding_resnet101_cityscapes_131415.pth
```
