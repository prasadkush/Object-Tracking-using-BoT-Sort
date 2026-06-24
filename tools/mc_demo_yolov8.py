import argparse
import time
from pathlib import Path
import sys


import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import numpy as np
from ultralytics import YOLO
from util import COLORS_10, draw_bboxes

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

sys.path.insert(0, './yolov7')
sys.path.append('.')

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))


def _get_features(bbox_xywh, ori_img):
    im_crops = []
    for box in bbox_xywh:
        x1,y1,x2,y2 = self._xywh_to_xyxy(box)
        im = ori_img[y1:y2,x1:x2]
        im_crops.append(im)
    if im_crops:
        features = self.extractor(im_crops)
    else:
        features = np.array([])
    return features

def xywh_to_tlbr(bbox_xywh, orig_img):
    bbox_tlbr = torch.zeros((bbox_xywh.shape[0],4))
    bbox_tlbr[:,0:2] = torch.floor(torch.clamp(bbox_xywh[:,0:2] - bbox_xywh[:,2:4]/2, min = 0))
    bbox_tlbr[:,2:4] = torch.ceil(torch.clamp(bbox_xywh[:,0:2] + bbox_xywh[:,2:4]/2, max = torch.tensor([orig_img.shape[0]-1, orig_img.shape[1]-1])))
    return bbox_tlbr


def tlwh_to_tlbr(bbox_tlwh, orig_img):
    bbox_tlbr = torch.zeros((bbox_tlwh.shape[0],4))
    bbox_tlbr[:,0:2] = torch.floor(torch.clamp(bbox_tlwh[:,0:2], min = 0))
    bbox_tlbr[:,2:4] = torch.ceil(torch.clamp(bbox_tlwh[:,0:2] + bbox_tlwh[:,2:4], max = torch.tensor([orig_img.shape[0]-1, orig_img.shape[1]-1])))
    return bbox_tlbr




def detect(args, save_img=False):
    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    #save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    #webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #    ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    #set_logging()
    device = select_device(args.device)
    half = device.type != 'cpu'  # half
    #stride = int(model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size
    print('os.pardir: ', os.path.abspath(os.pardir))
    model = YOLO('/home/kush/Desktop/code/object_tracking/fruit_crop/runs/detect/fruits_3_train6/train/weights/best.pt')
    dirname = args.imgs_path
    print('dirname: ', dirname)

    # Second-stage classifier
    #classify = False
    #if classify:
    #    modelc = load_classifier(name='resnet101', n=2)  # initialize
    #    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    #vid_path, vid_writer = None, None
    #if webcam:
    #    view_img = check_imshow()
    #    cudnn.benchmark = True  # set True to speed up constant image size inference
    #    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    #else:
    #    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    #names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = BoTSORT(args, frame_rate=30.0)

    # Run inference
    #if device.type != 'cpu':
    #    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    filelist = os.listdir(dirname)
    filelist.sort()
    print('self.dirname: ', dirname)
    print('filelist: ', filelist)
            
    img_array = []
    for filename in filelist:
        imgpath = os.path.join(dirname, filename)
        print('imgpath: ', imgpath)

        im = cv2.imread(imgpath)
        im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)
        #img = torch.from_numpy(img).to(device)
        #img = img.half() if half else img.float()  # uint8 to fp16/32
        #img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #if img.ndimension() == 3:
        #    img = img.unsqueeze(0)
            
        # Inference
        t1 = time_synchronized()
        #pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #t2 = time_synchronized()

        # Apply Classifier
        #if classify:
        #    pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        results = []

        result = model(im)  

        bbox_xcycwh = result[0].boxes.xywh.to(torch.int32)   # convert to int  
        cls_id = result[0].boxes.cls.to(torch.int32)        #convert to int
        cls_conf = result[0].boxes.conf



        #features = _get_features(bbox_xcycwh, ori_img)
        bbox_tlbr = xywh_to_tlbr(bbox_xcycwh, im)
        #detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        print('cls_conf.shape: ', cls_conf.shape)
        print('cls_id.shape: ', cls_id.shape)
        detections = torch.cat((bbox_tlbr, torch.reshape(cls_conf, (cls_conf.shape[0], 1)), torch.reshape(cls_id, (cls_id.shape[0], 1))), dim=1)
        detections = detections.numpy()
        #print('bbox_tlbr: ', bbox_tlbr)
        #print('cls_conf: ', cls_conf)
        #print('cls_id: ', cls_id)
        #print('detections: ', detections)
        imc = im.copy()
        for j in range(bbox_tlbr.shape[0]):
            cv2.rectangle(imc,(int(bbox_tlbr[j,0]), int(bbox_tlbr[j,1])), (int(bbox_tlbr[j,2]), int(bbox_tlbr[j,3])),(255,0,0),2)
        cv2.imshow('im', imc)

            # Run tracker
            #detections = []
            #if len(det):
            #    boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
            #    boxes = boxes.cpu().numpy()
            #    detections = det.cpu().numpy()
            #    detections[:, :4] = boxes
        imc2 = imc.copy()
        online_targets = tracker.update(detections, im, imc2.copy())

        print('len(online_targets: ', len(online_targets))

        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_cls = []
        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            if tlwh[2] * tlwh[3] > args.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_cls.append(t.cls)
        if len(online_tlwhs) > 0:
            bbox_xyxy = tlwh_to_tlbr(torch.tensor(online_tlwhs), im)
            bbox_xyxy = bbox_xyxy.tolist()
            im_boxes = draw_bboxes(im, bbox_xyxy, online_ids)
            #im_out = cv2.resize(im_boxes, (args.display_width, args.display_height), interpolation=cv2.INTER_LINEAR)
            im_out = cv2.resize(im_boxes, (880, 660), interpolation=cv2.INTER_LINEAR)
        else:
            #im_out = cv2.resize(im, (args.display_width, args.display_height), interpolation=cv2.INTER_LINEAR)
            im_out = cv2.resize(im_boxes, (880, 660), interpolation=cv2.INTER_LINEAR)
        #img_array.append(im_out)
        imc = cv2.resize(imc, (880, 660), interpolation=cv2.INTER_LINEAR)
        im_out2 = np.concatenate((imc, np.zeros((imc.shape[0],4,3)).astype(np.uint8), im_out), axis = 1)
        cv2.putText(im_out2, 'yolov8 output', (int(imc.shape[1]/2) - 60, 35), cv2.FONT_HERSHEY_PLAIN, 2.0, [255,0,0], 4)
        cv2.putText(im_out2, 'BoT-Sort output', (imc.shape[1] + int(im_out.shape[1]/2) - 60 , 35), cv2.FONT_HERSHEY_PLAIN, 2.0, [255,0,0], 4)
        img_array.append(im_out2)

        if args.display:
            img_array.append(im_out2)
            cv2.imshow("test", im_out)
            cv2.imshow("test2", im_out2)
            cv2.waitKey(1)

    videoname = 'video2.mp4'
    if args.VIDEO_PATH:
        size = (im_out2.shape[1], im_out2.shape[0])
        videofile = os.path.join(args.VIDEO_PATH, videoname)
        #out = cv2.VideoWriter(videofile,cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
        out = cv2.VideoWriter(videofile,cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        print('created video at path: ', videofile)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.25, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.05, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=0, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.1,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument("--VIDEO_PATH", type=str, default="/home/kush/Desktop/code/object_tracking/fruit_crop/BoT-SORT")
    #parser.add_argument("--display", dest='feature', action='store_false') # DOES NOT WORK YET
    #parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    #parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="False")
    parser.add_argument("--imgs_path", type=str, default="/home/kush/Desktop/code/object_tracking/fruit_crop/APPLE_MOTS/testing/images/0006")

    args = parser.parse_args()

    args.jde = False
    args.ablation = False

    # check_requirements(exclude=('pycocotools', 'thop'))

    detect(args)
