# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
from PIL import Image, ImageOps
import numpy as np
import json

import torch
from adet.layers import BezierAlign
from adet.structures import Beziers
from detectron2.layers import cat
from torch import nn
import cv2
import tqdm
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
import gc

from demo_bezier import Model

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def get_size(image_size, w, h):
    w_ratio = w / image_size[1]
    h_ratio = h / image_size[0]
    down_scale = max(w_ratio, h_ratio)
    if down_scale > 1:
        return down_scale
    else:
        return 1


def test_bezier(scale, path_img, path_img_bezier, cps, output_size):
    # image_size = (2560, 2560)  # H x W
    # output_size = (256, 1024)
    im = Image.open(path_img)
    w, h = im.size
    image_size = (h,w)

    input_size = (image_size[0] // scale,
                  image_size[1] // scale)
    m = Model(input_size, output_size, 1 / scale).cuda()

    beziers = [[]]
    im_arrs = []
    down_scales = []
    
    # imgfile = '1019.jpg'
    # im = Image.open('tools/tests/imgs/' + imgfile)
    im = Image.open(path_img)
    # im.show()
    # pad
    w, h = im.size
    down_scale = get_size(image_size, w, h)
    down_scales.append(down_scale)
    if down_scale > 1:
        im = im.resize((int(w / down_scale), int(h / down_scale)), Image.ANTIALIAS)
        w, h = im.size
    padding = (0, 0, image_size[1] - w, image_size[0] - h)
    im = ImageOps.expand(im, padding)
    im = im.resize((input_size[1], input_size[0]), Image.ANTIALIAS)
    im_arrs.append(np.array(im))

    # cps = [152.0, 209.0, 134.1, 34.18, 365.69, 66.2, 377.0, 206.0, 345.0, 214.0, 334.31, 109.71, 190.03, 80.12, 203.0, 214.0] # 1019
    # cps = [1193.1549,  374.1745, 1360.2979,  250.3801, 1353.9391,  254.4915,
    #      1515.5188,  119.4304, 1519.0569,  281.8831, 1357.5431,  421.2469,
    #      1356.0538,  416.1286, 1194.0586,  553.8478]


    # cps = np.array(cps)[[1, 0, 3, 2, 5, 4, 7, 6, 15, 14, 13, 12, 11, 10, 9, 8]]
    cps = np.array(cps)[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
    beziers[0].append(cps)
    beziers = [torch.from_numpy(np.stack(b)).cuda().float() for b in beziers]
    beziers = [b / d for b, d in zip(beziers, down_scales)]

    im_arrs = np.stack(im_arrs)
    x = torch.from_numpy(im_arrs).permute(0, 3, 1, 2).cuda().float()
    
    x = m(x, beziers)
    for i, roi in enumerate(x):
        roi = roi.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        im = Image.fromarray(roi, "RGB")
        im.save(path_img_bezier)
    loss = x.mean()
    loss.backward()
    # print(m)
    del m
    gc.collect()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    if not os.path.exists('output_bezier_2'):
        os.makedirs('output_bezier_2')
    if not os.path.exists('image_bezier_2'):
        os.makedirs('image_bezier_2')
    if not os.path.exists('sample_output_2'):
        os.makedirs('sample_output_2')
    cfg = setup_cfg(args)
    txt_dir = 'output_bezier_2/'
    output_bezier_dir = 'image_bezier_2/'
    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            print("path: ",path)
            name_img = os.path.basename(path)
            # name = name_img[:-4]
            f = open(txt_dir + name_img +'.txt', 'w+')
            # print(predictions)
            bezier = predictions["instances"].beziers
            # print(bezier[0])
            count = 1
            for bz in bezier:
                p = [0]*8
                x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8 = map(int, bz)
                p[0] = x1
                p[2] = x4
                p[4] = x5
                p[6] = x8

                p[1] = min(y1,y2)
                p[3] = min(y3,y4)
                p[5] = max(y5,y6)
                p[7] = max(y7,y8)
                img_bezier = name_img[:-4] +'_'+ str(count) + '.jpg'
                path_img_bezier = os.path.join(output_bezier_dir,img_bezier)
                bz = bz.tolist()
                count+=1
                out_w = abs(max(x4,x5) - min(x1,x8))
                out_h = abs(max(p[5],p[7]) - min(p[1],p[3]))
                if out_w == 0 or out_h == 0:
                    out_w = 1
                    out_h = 1
                output_size = (out_h, out_w)
                # print(bz)
                # print(output_size)
                test_bezier(1, path, path_img_bezier, bz, output_size)
                f.write("{},{},{},{},{},{},{},{},{}{}".format(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],img_bezier,"\n"))
            f.close()
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
