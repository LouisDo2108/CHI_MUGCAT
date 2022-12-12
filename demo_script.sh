eval "$(conda shell.bash hook)"
conda activate open-mmlab

# 2d_to_3d
# cd /root/data/ltnghia/projects/visual_communication/htluc/custom_code
# python mytest.py \
#     /root/data/ltnghia/projects/visual_communication/htluc//mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
#     /root/data/ltnghia/projects/visual_communication/htluc/custom_code/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k.pth \
#     /root/data/ltnghia/projects/visual_communication/htluc/mmpose/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py \
#     /root/data/ltnghia/projects/visual_communication/htluc/custom_code/res50_intehand3d_all_256x256.pth \
#     --json-file tests/data/interhand2.6m/test_interhand2.6m_data.json \
#     --img-root tests/data/interhand2.6m \
#     --out-img-root vis_results \
#     --rebase-keypoint-height \
#     --video-path /root/data/ltnghia/projects/visual_communication/htluc/custom_code/00421.mp4

cd /root/data/ltnghia/projects/visual_communication/htluc/CHI_MUGCAT

# 2d hand keypoints
python src/2d_hand_keypoints.py \
    /root/data/ltnghia/projects/visual_communication/htluc/mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    models/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k.pth \
    /root/data/ltnghia/projects/visual_communication/htluc/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_all_256x256.py \
    models/res50_interhand2d_256x256_all.pth \
    --video-path data/videos/00421.mp4 \

# 2d human pose
python src/2d_human_pose.py \
    /root/data/ltnghia/projects/visual_communication/htluc/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    models/faster_rcnn_r50_fpn_1x_coco.pth \
    /root/data/ltnghia/projects/visual_communication/htluc/mmpose/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py \
    models/hrnet_w48_posetrack18_384x288_posewarper_stage2.pth \
    --video-path data/videos/00421.mp4 \
    --use-multi-frames --online