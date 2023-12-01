# Gaze-Corrected EG3D

Enhanced EG3D model for realistic 3D face generation with improved gaze alignment.  
This version integrates gaze estimation to correct gaze bias, ensuring eye orientation is consistent with facial pose.

## Environment Setup
Refer to [Environment Setup Link] for installation and setup instructions.

## Model
- Pretrained model checkpoint available at [Checkpoint Link].
- Official EG3D and L2CS checkpoints can be downloaded from [Official Links].

## Training Method
```bash
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/data/ --resume=~/checkpoint.pkl --gpus=4 --batch=16 --gamma=1 --batch-gpu=4 --gen_pose_cond=True --neural_rendering_resolution_final=128
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
```bash
python calc_gfas.py --network=~/checkpoint.pkl
```

For other evaluation methods like FID, KID, etc., refer to the original EG3D GitHub [Link].
