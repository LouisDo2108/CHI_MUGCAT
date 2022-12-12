# Installation

1. Create a conda environemnt:
```
conda create -n open-mmlab python=3.7
```

2. Install pytorch with cudatoolkit's version matches with your GPUs' CUDA version.

3. Install mmdetection: https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation

4. Install mmpose https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md#installation

5. Install requirements.txt

6. Download the pretrained models from [here](https://www.google.com).

# How to use

cd to the code's root directory. Modify the video paths in demo_script.sh and run
```
bash demo_script.sh
```

The human pose keypoints JSONs and hand keypoints JSONs are in output_human_pose and  ./output_hand_keypoints, respectively.