# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import numpy as np
import json
from pathlib import Path
import torch
import torchvision.ops.boxes as bops

import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# import the necessary packages
import numpy as np
# Malisiewicz et al.
def non_max_suppression_fast(pose_results, overlapThresh=0.7):
    
    boxes = []
    for pose_result in pose_results:
        boxes.append(pose_result['bbox'])
    # boxes = np.array(boxes)
    ious = {}
    for ix, x in enumerate(boxes):
        for iy, y in enumerate(boxes):
            if iy > ix:
                x_ = torch.tensor([x[:-1]], dtype=torch.float)
                y_ = torch.tensor([y[:-1]], dtype=torch.float)
                ious[(ix, iy)] = bops.box_iou(x_, y_)
                # print(ix, iy, ious[(ix, iy)])
    ious = sorted(ious, key=ious.get)
    # print("Sorted:", ious)
    pick = list(ious[0])
    # print("Pick:", pick)
    # # if there are no boxes, return an empty list
    # if len(boxes) == 0:
    #     return []
	# # if the bounding boxes integers, convert them to floats --
	# # this is important since we'll be doing a bunch of divisions
    # if boxes.dtype.kind == "i":
    #     boxes = boxes.astype("float")
	# # initialize the list of picked indexes	
    # pick = []
	# # grab the coordinates of the bounding boxes
    # x1 = boxes[:,0]
    # y1 = boxes[:,1]
    # x2 = boxes[:,2]
    # y2 = boxes[:,3]
	# # compute the area of the bounding boxes and sort the bounding
	# # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # idxs = np.argsort(y2)
	# # keep looping while some indexes still remain in the indexes
	# # list
    # while len(idxs) > 0:
	# 	# grab the last index in the indexes list and add the
	# 	# index value to the list of picked indexes
    #     last = len(idxs) - 1
    #     i = idxs[last]
    #     pick.append(i)
	# 	# find the largest (x, y) coordinates for the start of
	# 	# the bounding box and the smallest (x, y) coordinates
	# 	# for the end of the bounding box
    #     xx1 = np.maximum(x1[i], x1[idxs[:last]])
    #     yy1 = np.maximum(y1[i], y1[idxs[:last]])
    #     xx2 = np.minimum(x2[i], x2[idxs[:last]])
    #     yy2 = np.minimum(y2[i], y2[idxs[:last]])
	# 	# compute the width and height of the bounding box
    #     w = np.maximum(0, xx2 - xx1 + 1)
    #     h = np.maximum(0, yy2 - yy1 + 1)
	# 	# compute the ratio of overlap
    #     overlap = (w * h) / area[idxs[:last]]
	# 	# delete all indexes from the index list that have
    #     idxs = np.delete(idxs, np.concatenate(([last],
	# 		np.where(overlap > overlapThresh)[0])))
	# # return only the bounding boxes that were picked using the
	# # integer data type
    # # print(pick)
    # # return boxes[pick].astype("int")
    pose_results = [pose_results[i] for ix, i in enumerate(pick) if ix < 2]
    return pose_results

def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
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

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file
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

    
    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # Create json output folder
    output_path = Path("./output_hand_keypoints/", Path(args.video_path).stem)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, cur_frame)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        # Iterate over multiple bbox 
        print()
        print("Before:", len(pose_results))
        if len(pose_results) > 2:
            pose_results = non_max_suppression_fast(pose_results)
        print("After:", len(pose_results))
        
        json_list = []
        for x in pose_results:
            
            # Jsonify bbox numpy arrays
            for k, v in x.items():
                if isinstance(v, np.ndarray):
                    x[k] = v.tolist()
            
            # Add frame_id
            x['frame_id'] = frame_id
            json_list.append(x)
        
        with open(str(output_path.resolve() / str(frame_id)) + '.json', 'w') as fout:
            json.dump(json_list , fout)

        # show the results
        vis_frame = vis_pose_result(
            pose_model,
            cur_frame,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False)
        # cv2.imwrite("/root/data/ltnghia/projects/visual_communication/htluc/CHI_MUGCAT/out.png", vis_frame)
        # break

        if args.show:
            cv2.imshow('Frame', vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
