# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from natsort import natsorted
from argparse import ArgumentParser
from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import mmcv
import numpy as np
from xtcocotools.coco import COCO

from mmpose.apis import inference_interhand_3d_model, vis_3d_pose_result
from mmpose.apis.inference import init_pose_model
from mmpose.core import SimpleCamera


def _transform_interhand_camera_param(interhand_camera_param):
    """Transform the camera parameters in interhand2.6m dataset to the format
    of SimpleCamera.

    Args:
        interhand_camera_param (dict): camera parameters including:
            - camrot: 3x3, camera rotation matrix (world-to-camera)
            - campos: 3x1, camera location in world space
            - focal: 2x1, camera focal length
            - princpt: 2x1, camera center

    Returns:
        param (dict): camera parameters including:
            - R: 3x3, camera rotation matrix (camera-to-world)
            - T: 3x1, camera translation (camera-to-world)
            - f: 2x1, camera focal length
            - c: 2x1, camera center
    """
    camera_param = {}
    camera_param['R'] = np.array(interhand_camera_param['camrot']).T
    camera_param['T'] = np.array(interhand_camera_param['campos'])[:, None]
    camera_param['f'] = np.array(interhand_camera_param['focal'])[:, None]
    camera_param['c'] = np.array(interhand_camera_param['princpt'])[:, None]
    return camera_param


def main():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose network')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--camera-param-file',
        type=str,
        default=None,
        help='Camera parameter file for converting 3D pose predictions from '
        ' the pixel space to camera space. If None, keypoints in pixel space'
        'will be visualized')
    parser.add_argument(
        '--gt-joints-file',
        type=str,
        default=None,
        help='Optional argument. Ground truth 3D keypoint parameter file. '
        'If None, gt keypoints will not be shown and keypoints in pixel '
        'space will be visualized.')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--show-ground-truth',
        action='store_true',
        help='If True, show ground truth keypoint if it is available.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default=None,
        help='Root of the output visualization images. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument('--video-path', type=str, help='Video path')

    args = parser.parse_args()
    assert args.show or (args.out_img_root != '')
    
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']
    
    
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    video = mmcv.VideoReader(args.video_path)
    
    det_results_list = []
    
    # for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
    img_dir = "/root/data/ltnghia/projects/visual_communication/htluc/custom_code/00421"
    # pose_model_ = init_pose_model(
    #     "/root/data/ltnghia/projects/visual_communication/htluc/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py", 
    #     "/root/data/ltnghia/projects/visual_communication/htluc/custom_code/res50_onehand10k_256x256.pth",
    #     device=args.device.lower())

    # dataset_ = pose_model_.cfg.data['test']['type']
    # dataset_info_ = pose_model_.cfg.data['test'].get('dataset_info', None)
    
    # if dataset_info_ is None:
    #     warnings.warn(
    #         'Please set `dataset_info` in the config.'
    #         'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
    #         DeprecationWarning)
    # else:
    #     dataset_info_ = DatasetInfo(dataset_info_)
    
    for frame_id, cur_frame in enumerate(natsorted(os.listdir(img_dir))):
        
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        # cur_frame = mmcv.image.imread("/root/data/ltnghia/projects/visual_communication/htluc/mmpose/tests/data/interhand2.6m/image2017.jpg")
        image_name = os.path.join(img_dir, cur_frame)
        cur_frame = mmcv.image.imread(image_name)
        mmdet_results = inference_detector(det_model, cur_frame)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, 1)
        
        # # optional
        # return_heatmap = False

        # # e.g. use ('backbone', ) to return backbone feature
        # output_layer_names = None


        # pose_results, returned_outputs = inference_top_down_pose_model(
        #     pose_model_,
        #     image_name,
        #     person_results,
        #     bbox_thr=0.3,
        #     format='xyxy',
        #     dataset=dataset_,
        #     dataset_info=dataset_info_,
        #     return_heatmap=False,
        #     outputs=None
        # )
        # print(pose_results)
        
        for x in person_results:
            bbox = x['bbox']
            det_results = [{
                'image_name': image_name,
                'bbox': bbox,  # bbox format is 'xywh'
                'camera_param': None,
                'keypoints_3d_gt': None
            }]
            det_results_list.append(det_results)
        # break

    for i, det_results in enumerate(
            mmcv.track_iter_progress(det_results_list)):

        image_name = det_results[0]['image_name']
        pose_results = inference_interhand_3d_model(
            pose_model, cur_frame, det_results, dataset=dataset, format='xyxy')

        # Post processing
        pose_results_vis = []
        for idx, res in enumerate(pose_results):
            keypoints_3d = res['keypoints_3d']
            # normalize kpt score
            if keypoints_3d[:, 3].max() > 1:
                keypoints_3d[:, 3] /= 255
            # get 2D keypoints in pixel space
            res['keypoints'] = keypoints_3d[:, [0, 1, 3]]

            # For model-predicted keypoints, channel 0 and 1 are coordinates
            # in pixel space, and channel 2 is the depth (in mm) relative
            # to root joints.
            # If both camera parameter and absolute depth of root joints are
            # provided, we can transform keypoint to camera space for better
            # visualization.
            camera_param = res['camera_param']
            keypoints_3d_gt = res['keypoints_3d_gt']
            if camera_param is not None and keypoints_3d_gt is not None:
                # build camera model
                camera = SimpleCamera(camera_param)
                # transform gt joints from world space to camera space
                keypoints_3d_gt[:, :3] = camera.world_to_camera(
                    keypoints_3d_gt[:, :3])

                # transform relative depth to absolute depth
                keypoints_3d[:21, 2] += keypoints_3d_gt[20, 2]
                keypoints_3d[21:, 2] += keypoints_3d_gt[41, 2]

                # transform keypoints from pixel space to camera space
                keypoints_3d[:, :3] = camera.pixel_to_camera(
                    keypoints_3d[:, :3])

            # rotate the keypoint to make z-axis correspondent to height
            # for better visualization
            vis_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            keypoints_3d[:, :3] = keypoints_3d[:, :3] @ vis_R
            if keypoints_3d_gt is not None:
                keypoints_3d_gt[:, :3] = keypoints_3d_gt[:, :3] @ vis_R

            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                valid = keypoints_3d[..., 3] > 0
                try:
                    keypoints_3d[..., 2] -= np.min(
                        keypoints_3d[valid, 2], axis=-1, keepdims=True)
                except Exception as e:
                    print("Failed to rebase")
            res['keypoints_3d'] = keypoints_3d
            res['keypoints_3d_gt'] = keypoints_3d_gt
            # print(res['keypoints_3d'])
            # Add title
            instance_id = res.get('track_id', idx)
            res['title'] = f'Prediction ({instance_id})'
            pose_results_vis.append(res)
            print(res)
            # # Add ground truth
            # if args.show_ground_truth:
            #     if keypoints_3d_gt is None:
            #         print('Fail to show ground truth. Please make sure that'
            #               ' gt-joints-file is provided.')
            #     else:
            #         gt = res.copy()
            #         if args.rebase_keypoint_height:
            #             valid = keypoints_3d_gt[..., 3] > 0
            #             keypoints_3d_gt[..., 2] -= np.min(
            #                 keypoints_3d_gt[valid, 2], axis=-1, keepdims=True)
            #         gt['keypoints_3d'] = keypoints_3d_gt
            #         gt['title'] = f'Ground truth ({instance_id})'
            #         pose_results_vis.append(gt)

        # Visualization
        if args.out_img_root is None:
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = osp.join(args.out_img_root, f'vis_{image_name}.jpg')

        vis_3d_pose_result(
            pose_model,
            result=pose_results_vis,
            img=det_results[0]['image_name'],
            out_file=out_file,
            dataset=dataset,
            show=args.show,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            axis_azimuth=-115,
        )


if __name__ == '__main__':
    main()
