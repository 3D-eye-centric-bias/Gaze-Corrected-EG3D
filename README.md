# Gaze-Corrected EG3D

Enhanced EG3D model for realistic 3D face generation with improved gaze alignment.  
This version integrates gaze estimation to correct gaze bias, ensuring eye orientation is consistent with facial pose.

## Environment Setup
Refer to [Environment Setup Guide](https://github.com/3D-eye-centric-bias/Gaze-Corrected-EG3D/blob/main/docs/env_guide.md) for installation and setup instructions.

## Model
- Model checkpoint available at [Checkpoint Link](https://drive.google.com/drive/folders/1Bl__aGhCtGBXNSnpAaTAsozxzkykuFjb?usp=sharing).
- Official EG3D and L2CS checkpoints can be downloaded from
  
  1. [EG3D checkpoint link](https://github.com/NVlabs/eg3d/blob/main/docs/models.md)
  
  2. [L2CS checkpoint link](https://drive.google.com/drive/folders/1qDzyzXO6iaYIMDJDSyfKeqBx8O74mF8s)

## Training
Specify the paths to your **data.zip** and **checkpoint.pkl** for the data and resume arguments.
```bash
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/data.zip --resume=~/checkpoint.pkl --gpus=4 --batch=16 --gamma=1 --batch-gpu=4 --gen_pose_cond=True --neural_rendering_resolution_final=128
```

## Image Generation
1. Single Network
```bash
python gen_samples.py --network=~/checkpoint.pkl --outdir=out/ --seeds=0-3
```

2. Comparing Two Networks
```bash
python gen_samples_gaze_compare.py --network=~/checkpoint1.pkl --network2=~/checkpoint2.pkl --outdir=out/ --seeds=0-3
```

## Evaluation Method
**GFAS Score**
```bash
python calc_gfas.py --network=~/checkpoint.pkl
```
**FID**
**KID**

## Citation
```bash
@inproceedings{Chan2022,
  author = {Eric R. Chan and Connor Z. Lin and Matthew A. Chan and Koki Nagano and Boxiao Pan and Shalini De Mello and Orazio Gallo and Leonidas Guibas and Jonathan Tremblay and Sameh Khamis and Tero Karras and Gordon Wetzstein},
  title = {Efficient Geometry-aware {3D} Generative Adversarial Networks},
  booktitle = {CVPR},
  year = {2022}
}
```

## Acknowledge
This project is built on source codes shared by [EG3D](https://github.com/NVlabs/eg3d) and [L2CS](https://github.com/Ahmednull/L2CS-Net)
