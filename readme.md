# Installation

1. Create a conda environemnt and activate the environment
```
conda create -n open-mmlab python=3.7
```

2. After creating the environment, activate it
```
conda activate open-mmlab
```

3. Install pytorch with cudatoolkit's version matches with your GPUs' CUDA version.

4. Install mmdetection: https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation

5. Install mmpose https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md#installation

6. Install requirements.txt
```
pip install -r requirements.txt
```

7. Download the pretrained models from [here](https://www.google.com).

# How to use

cd to CHI_MUGCAT's root directory. Modify the video paths in demo_script.sh and run
```
bash demo_script.sh
```

The human pose keypoints JSONs and hand keypoints JSONs are in output_human_pose and  ./output_hand_keypoints, respectively.