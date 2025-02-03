import os
import cv2
import sys
import time
import torch
import argparse
import importlib

from loguru import logger
from utils import Bbox2d
from byte_tracker import ByteTracker
from yolox.data.datasets import COCO_CLASSES
from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def parse_args():
    parser = argparse.ArgumentParser("tracker demo")
    parser.add_argument("--name", "-n", type=str, default="yolo_tiny")
    parser.add_argument("--video", "-v", type=str)
    parser.add_argument("--ckpt", "-p", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--lconf", default=0.15, type=float, help="lower conf")
    parser.add_argument("--hconf", default=0.45, type=float, help="higher conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    return parser.parse_args()


def get_exp_by_name(exp_name):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", "default", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=True,
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35, visualize=False):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]  # top left && bottom right
        # preprocessing: resize
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        if visualize:
            vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        else:
            vis_res = img
        return vis_res, bboxes, scores


def video_parser(video_path: str):
    cap = cv2.VideoCapture(video_path)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            yield frame
        else:
            break


def main(exp, args):
    if args.lconf is not None:
        exp.test_conf = args.lconf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    tracker = ByteTracker(args.lconf, args.hconf)

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    # get model
    model = exp.get_model()
    model.eval()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    ckpt_file = args.ckpt
    logger.info(f"loading checkpoint {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    predictor = Predictor(
        model,
        exp,
        COCO_CLASSES,
        device="cpu",
    )
    current_time = time.localtime()
    for frame in video_parser(args.video):
        outputs, img_info = predictor.inference(frame)
        result_frame, bboxes, scores = predictor.visual(
            outputs[0], img_info, predictor.confthre
        )
        tracker.update(bboxes, scores)
        box_img = tracker.visualize_tracklets(frame)
        cv2.imshow("result", box_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

if __name__ == "__main__":
    args = parse_args()
    exp = get_exp_by_name(args.name)
    main(exp, args)
