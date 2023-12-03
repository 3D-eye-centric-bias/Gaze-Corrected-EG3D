# Gaze-Corrected EG3D: Overcoming Camera-Facing Gaze Bias in EG3D Scene Generation

### Implementation of Gaze-Corrected EG3D - Enhanced EG3D model with improved gaze alignment

<img width="786" alt="qualitative" src="https://github.com/3D-eye-centric-bias/Gaze-Corrected-EG3D/assets/89647814/5d01fb2d-7d80-4398-8942-8406d2ac5ad9">

You can try to generate above image with seed number 346.

Abstract: Notwithstanding that EG3D achieved realistic image quality with high computational efficiency, the gaze following problem hinders the real-world application of 3D GANs such as virtual reality and human-computer interaction. Due to the data bias when rotating the face, all existing 3D GANs cannot maintain the direction of eye when the camera is rotating and gazes the camera. In this work, we tackle this problem by fusing 3D-aware image synthesis with gaze estimation for the first time. We subjoin a pretrained state-of-the-art gaze estimation model called L2CS-Net into training pipeline of EG3D, successfully addressed the inherent gaze following issue on FFHQ dataset.
 
 
## Quick Setup
Refer to [Environment Setup Guide](https://github.com/3D-eye-centric-bias/Gaze-Corrected-EG3D/blob/main/docs/env_guide.md) for installation and setup instructions.

## Download Models
- Our trained model is available [here](https://drive.google.com/file/d/1LAzjJBhp5GGZLymWt_VP-LMzOHI8pfzj/view?usp=sharing) (360MB).
- Original EG3D models are available [here](https://github.com/NVlabs/eg3d/blob/main/docs/models.md).  
  (We used ffhq512-128.pkl as a baseline)

## Data Access
We provide dataset for the experiments:
- Training: [Download](https://drive.google.com/file/d/1pFl0gWlhMIEKKfgLp3abIKD6DNPVMJ_x/view?usp=sharing) (8.9GB)
- Evaluation: [Download](https://drive.google.com/file/d/1bkdSXkc8UHhRyWiIdafUUqRLpwryqjQz/view?usp=sharing) (27GB)  

After downloading the zip files, you need to **unzip** them.

## Training
Set your paths and start training:
```bash
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/extracted/140 --resume=~/model.pkl --l2cs-path=~/L2CSNet_gaze360.pkl --gpus=4 --batch=16 --gamma=1 --batch-gpu=4 --gen_pose_cond=True --neural_rendering_resolution_final=128
```

## Generate Images
1. Single Network
```bash
python gen_samples.py --network=~/model.pkl --outdir=out --seeds=0-3
```

2. Comparing Two Networks
```bash
python gen_samples_compare.py --network=~/model1.pkl --network2=~/model2.pkl --l2cs-path=~/L2CSNet_gaze360.pkl --outdir=out --seeds=0-3
```

3. Generate images and sort based on GFAS score
```bash
python gen_good_samples.py --network=~/model.pkl --l2cs-path=~/L2CSNet_gaze360.pkl --outdir=out
```

## Evaluation  
You can evaluate the trained model :

**GFAS, FID, KID, ArcFace Evaluation**
```bash
cd eg3d

#GFAS Score (Gaze-Face Alignment Score)
python calc_gfas.py --network=~/model.pkl --l2cs-path=~/L2CSNet_gaze360.pkl

#FID(Frechet Inception Distance)
python calc_metrics.py --network=~/model.pkl --metrics=fid50k_full --data=~/eval

#KID(Kernel Inception Distance)
python calc_metrics.py --network=~/model.pkl --metrics=kid50k_full --data=~/eval

#ArcFace(Identity consistency)
python calc_id.py --network=~/model.pkl --outdir=out
```

**Pose(Pose Accuracy)**
1. Generate images with base camera parameters.
```bash
#eg3d/
python gen_samples_pose.py --network=~/model.pkl --outdir=pose
```
2. Extract camera parameters from images.
```bash
#data_preprocessing/ffhq/
cd ../data_preprocessing/ffhq/
python preprocess_in_the_wild.py --indir=../../eg3d/pose
```
3. Calculate L2 distance between base and extracted camera parameters.
```bash
#eg3d/
cd ../../eg3d
python calc_pose.py --file1=pose/labels_generate.json --file2=../dataset_preprocessing/ffhq/pose/dataset.json
```

Note that evaluation time for FID, KID, Pose, and ArcFace are spent around 30 minutes. GFAS computation is not that expensive, it requires around 5 minutes to evaluate.

## Citation
```bash
@inproceedings{Chan2022,
  author = {Eric R. Chan and Connor Z. Lin and Matthew A. Chan and Koki Nagano and Boxiao Pan and Shalini De Mello and Orazio Gallo and Leonidas Guibas and Jonathan Tremblay and Sameh Khamis and Tero Karras and Gordon Wetzstein},
  title = {Efficient Geometry-aware {3D} Generative Adversarial Networks},
  booktitle = {CVPR},
  year = {2022}
}
@inproceedings{Ahmednull,
    title={L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments},
    author={Ahmed A.Abdelrahman and Thorsten Hempel and Aly Khalifa and Ayoub Al-Hamadi},
    booktitle={IEEE International Conference on Image Processing},
    year={2022}
}
@inproceedings{deng2019accurate,
    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2019}
}
```


## Acknowledgments
This project is built on source codes shared by [EG3D](https://github.com/NVlabs/eg3d), [L2CS](https://github.com/Ahmednull/L2CS-Net), [Deep3dFaceRecon_Pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
