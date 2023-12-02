# 🌟 Gaze-Corrected EG3D: Overcoming Camera-Facing Gaze Bias in EG3D Scene Generation

### ****Welcome** to Gaze-Corrected EG3D - **Enhanced EG3D model** with **improved gaze alignment!****

![image](https://github.com/3D-eye-centric-bias/Gaze-Corrected-EG3D/assets/63842546/ec4959b1-5e5c-44ee-9c3d-175a5bc4818c)


## Quick Setup
Refer to [Environment Setup Guide](https://github.com/3D-eye-centric-bias/Gaze-Corrected-EG3D/blob/main/docs/env_guide.md) for installation and setup instructions.

## Download Models
- Our trained model is available [here](https://drive.google.com/file/d/1LAzjJBhp5GGZLymWt_VP-LMzOHI8pfzj/view?usp=sharing).
- Original EG3D models are available [here](https://github.com/NVlabs/eg3d/blob/main/docs/models.md).  
  (We used ffhq512-128.pkl as a baseline)

## Data Access
We provide a sample dataset for quick experiments:
- Training: [Download](https://drive.google.com/file/d/1fhgC6hBY8_cnaMOR-TiGL5Pnp2djk8qb/view?usp=sharing)
- Evaluation: [Download](https://drive.google.com/file/d/1A6_MHbBt2sxUUHu7VFBFmMo2uoWsL8uh/view?usp=sharing)

## Training
Set your paths and start training:
```bash
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/data.zip --resume=~/checkpoint.pkl --gpus=4 --batch=16 --gamma=1 --batch-gpu=4 --gen_pose_cond=True --neural_rendering_resolution_final=128
```

## Generate Images
1. Single Network
```bash
python gen_samples.py --network=~/checkpoint.pkl --outdir=out/ --seeds=0-3
```

2. Comparing Two Networks
```bash
python gen_samples_gaze_compare.py --network=~/checkpoint1.pkl --network2=~/checkpoint2.pkl --outdir=out/ --seeds=0-3
```

## Evaluation  
**GFAS Score(Gaze-Face Alignment Score)**
```bash
python calc_gfas.py --network=~/checkpoint.pkl
```
**FID(Frechet Inception Distance)**
```bash
#eg3d/
python calc_metrics.py --network=~/checkpoint.pkl --metrics=fid50k_full --data=~/eval
```

**KID(Kernel Inception Distance)**
```bash
#eg3d/
python calc_metrics.py --network=~/checkpoint.pkl --metrics=kid50k_full --data=~/eval
```

**ID(Identity consistency)**
```bash
#eg3d/
python calc_id.py --network=~/checkpoint.pkl --outdir=out 
```

**Pose(Pose Accuracy)**
1. Generate images with base camera parameters.
```bash
#eg3d/
python gen_samples_pose.py --network=~/checkpoint.pkl --outdir=pose
```
2. Extract camera parameters from images.
```bash
#data_preprocessing/ffhq/
python preprocess_in_the_wild.py --indir=../../eg3d/pose
```
3. Calculate L2 distance between base and extracted camera parameters.
```bash
#eg3d/
python calc_pose.py --file1=pose/labels_generate.json --file2=../dataset_preprocessing/ffhq/pose/dataset.json
```

## Citation
```bash
@inproceedings{Chan2022,
  author = {Eric R. Chan and Connor Z. Lin and Matthew A. Chan and Koki Nagano and Boxiao Pan and Shalini De Mello and Orazio Gallo and Leonidas Guibas and Jonathan Tremblay and Sameh Khamis and Tero Karras and Gordon Wetzstein},
  title = {Efficient Geometry-aware {3D} Generative Adversarial Networks},
  booktitle = {CVPR},
  year = {2022}
}
```
```bash
@inproceedings{Ahmednull,
    title={L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments},
    author={Ahmed A.Abdelrahman and Thorsten Hempel and Aly Khalifa and Ayoub Al-Hamadi},
    booktitle={IEEE International Conference on Image Processing},
    year={2022}
}
```
```bash
@inproceedings{deng2019accurate,
    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2019}
}
```


## Acknowledgments
This project is built on source codes shared by [EG3D](https://github.com/NVlabs/eg3d), [L2CS](https://github.com/Ahmednull/L2CS-Net), [Deep3dFaceRecon_Pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
