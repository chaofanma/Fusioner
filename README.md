# Open-vocabulary Semantic Segmentation with Frozen Vision-Language Models

The repo contains official PyTorch implementation of BMVC 2022 oral paper **Open-vocabulary Semantic Segmentation with Frozen Vision-Language Models** by Chaofan Ma*, Yuhuan Yang*, Yanfeng Wang, Ya Zhang, and Weidi Xie.

![architecture](assets/framework.png)

For more information, check out the [project page](https://yyh-rain-song.github.io/Fusioner_webpage/) and the paper on [arXiv](https://arxiv.org/abs/2210.15138).


## Requirements

- python==3.7.11
- torch==1.9.0
- torchvision==0.10.0
- clip (from https://github.com/openai/CLIP)
- einops==0.3.2
- timm==0.4.12
- albumentations==1.1.0
- opencv-python==4.5.5.64


## Data Preparation
Same as [LSeg](https://github.com/isl-org/lang-seg), we follow [HSNet](https://github.com/juhongm999/hsnet) for data preparation. 
The datasets should be appropriately placed to have following directory structure:

For PASCAL-$5^i$ dataset:
```
dataset_root
    ├── SegmentationClassAug
    └── VOCdevkit
        └── VOC2012
            ├── Annotations
            ├── ImageSets
            ├── JPEGImages
            ├── SegmentationClass
            └── SegmentationObject

```
For COCO-$20^i$ dataset:
```
dataset_root
    ├── annotation
    ├── train2014
    └── val2014
```
More details such as datasets downloading please refers to [HSNet datasets preparing](https://github.com/juhongm999/hsnet#preparing-few-shot-segmentation-datasets).


## Training

```bash
python train.py --dataset_name {pascal, coco} \
                --dataset_root your/pascal/or/coco/dataset_root \
                --fold {0, 1, 2, 3} 
```


## Evaluation

```bash
python test.py --dataset_name {pascal, coco} \
               --dataset_root your/pascal/or/coco/dataset_root \
               --fold {0, 1, 2, 3} \
               --test_with_org_resolution \
               --load_ckpt_path path/to/saved/checkpoint
```

Currently, we do not add code about model saving when training, write it by yourself then pass through `--load_ckpt_path` for evaluation.


## Citation

If this code is useful for your research, please consider citing:

```bibtex
@inproceedings{ma2022fusioner,
  title     = {Open-vocabulary Semantic Segmentation with Frozen Vision-Language Models},
  author    = {Chaofan Ma, Yuhuan Yang, YanFeng Wang, Ya Zhang and Weidi Xie},
  booktitle = {British Machine Vision Conference},
  year      = {2022}
}
```


## Acknowledgements
Many thanks to the code bases from [LSeg](https://github.com/isl-org/lang-seg), [CLIP](https://github.com/openai/CLIP), [Segmenter](https://github.com/rstrudel/segmenter), [HSNet](https://github.com/juhongm999/hsnet).


