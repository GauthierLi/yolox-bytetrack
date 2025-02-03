import os
import cv2
import sys
from cv2.gapi.ot import TRACKED
import numpy as np

sys.path.append(os.path.dirname(__file__))
import torch

from typing import List
from loguru import logger
from utils import Bbox2d, get_cost_matrix
from scipy.optimize import linear_sum_assignment

class TrackStatus:
    NEW = 0
    TRACKED = 1
    MISSED = 2
    LOST = 3

debug_str_map = {
    TrackStatus.NEW:"NEW",
    TrackStatus.TRACKED:"TRACKED",
    TrackStatus.MISSED:"MISSED",
    TrackStatus.LOST:"LOST"
}

class Tracklet(object):
    CUR_IDX = 0
    def __init__(self,
                 idx: int,
                 location: Bbox2d,
                 max_history: int = 10) -> None:
        self.location = location
        self.status = TrackStatus.NEW
        self.idx = idx
        self.track_time = 0
        self.miss_time = 0
        self.lost_time = 0
        self.history = []

    def _update_location(self, bbox: Bbox2d=None): 
        if bbox is None:
            # TODO: kalman filter update
            ...
        else:
            self.location = bbox

    def set_tracked(self):
        self.track_time += 1 
        self.miss_time = 0 
        self.lost_time = 0
        if self.track_time > 3:
            self.status = TrackStatus.TRACKED

    def set_misses(self):
        self.status = TrackStatus.MISSED
        self.miss_time += 1 

    def set_lost(self):
        self.status = TrackStatus.LOST
        self.lost_time += 1


class ByteTracker(object):
    def __init__(self,
                 low_thr: float, 
                 high_thr: float,
                 max_miss_time: int = 5,
                 max_lost_time: int = 3) -> None:
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.tracklets: List[Tracklet] = [] 
        self.lost_tracklets: List[Tracklet] = [] # debug only
        self.max_miss_time = max_miss_time
        self.max_lost_time = max_lost_time

    def update(self, detections: torch.Tensor, scores: torch.Tensor):
        existed_bbox: List[Bbox2d] = [tc.location for tc in self.tracklets]
        high_conf_det_bbox: List[Bbox2d] = []
        low_conf_det_bbox: List[Bbox2d] = []
        for box, score in zip(detections.numpy(), scores.numpy()):
            bbox = Bbox2d(box[0], box[1], box[2], box[3])
            if score.item() > self.low_thr:
                if score.item() > self.high_thr:
                    high_conf_det_bbox.append(bbox)
                else:
                    low_conf_det_bbox.append(bbox)

        # matching detected bbox to tracklets
        logger.info(
            f"Start first high score matching step, lenth of detected box {len(high_conf_det_bbox)}, existed_bboxes {len(existed_bbox)}"
        )

        tmp_tracklets = []
        # 1. 1st match high score detecred box && existed box
        det_idxes, first_matched_idxes = self._match_step(
            high_conf_det_bbox, existed_bbox
        )

        first_not_matched_detections = []
        for i in range(len(high_conf_det_bbox)):
            if i not in det_idxes:
                first_not_matched_detections.append(high_conf_det_bbox[i])

        first_not_matched_tracklets = []
        for i in range(len(existed_bbox)):
            if i not in first_matched_idxes:
                first_not_matched_tracklets.append(self.tracklets[i])

        for det_i, match_i in zip(det_idxes, first_matched_idxes):
            self.tracklets[match_i]._update_location(high_conf_det_bbox[det_i])
            self.tracklets[match_i].set_tracked()
            tmp_tracklets.append(self.tracklets[match_i])

        for bbox in first_not_matched_detections:
            tmp_tracklet = Tracklet(Tracklet.CUR_IDX, bbox)
            Tracklet.CUR_IDX += 1
            tmp_tracklets.append(tmp_tracklet)


        # 2. 2nd match low score detecred box && 1st not_matched_tracklets box
        first_not_matched_bboxes = [tk.location for tk in first_not_matched_tracklets]
        second_det_idxes, second_matched_idxes = self._match_step(
            low_conf_det_bbox, first_not_matched_bboxes
        )

        for i in range(len(first_not_matched_bboxes)):
            if i not in second_matched_idxes:
                first_not_matched_tracklets[i]._update_location()
                first_not_matched_tracklets[i].set_misses()
                tmp_tracklets.append(first_not_matched_tracklets[i])

        for det_i, match_i in zip(second_det_idxes, second_matched_idxes):
            first_not_matched_tracklets[match_i]._update_location(low_conf_det_bbox[det_i])
            first_not_matched_tracklets[match_i].set_tracked()
            tmp_tracklets.append(first_not_matched_tracklets[match_i])

        self.tracklets.clear()
        tmp_tracklets.sort(key=lambda x: x.idx)
        for tracklet in tmp_tracklets:
            if tracklet.status == TrackStatus.MISSED and tracklet.miss_time > self.max_miss_time:
                tracklet.set_lost()
                self.lost_tracklets.append(tracklet)
                logger.debug(f"id: {tracklet.idx}, status: {debug_str_map[tracklet.status]}")
            else:
                logger.debug(f"id: {tracklet.idx}, status: {debug_str_map[tracklet.status]}")
                self.tracklets.append(tracklet)

        tmp_lost_tracklets = [] 
        for tracklet in self.lost_tracklets:
            if tracklet.lost_time < self.max_lost_time:
                tmp_lost_tracklets.append(tracklet)
        self.lost_tracklets = tmp_tracklets
        logger.info(f"high score detections: {len(high_conf_det_bbox)},\nfirst matched: {len(first_matched_idxes)},\nfirst not matched: {len(first_matched_idxes)},\nfirst not matched detections: {len(first_not_matched_bboxes)}"
            + f"\nlow score detections: {len(low_conf_det_bbox)}, second matched: {len(second_matched_idxes)}"
            + f"\ntracked or miss: {len(self.tracklets)}, lost: {len(self.lost_tracklets)}")

    def _match_step(self, existed_bboxes: List[Bbox2d], detected_bboxes: List[Bbox2d], thread: float = 0.8):
        cost_matrix = get_cost_matrix(existed_bboxes, detected_bboxes)
        # TODO: Get the implementation of linear_sum_assignment
        row_idxes, col_idxes = linear_sum_assignment(cost_matrix)
        post_row_idxes, post_col_idxes = [], []
        for row_idx, col_idx in zip(row_idxes, col_idxes):
            if cost_matrix[row_idx][col_idx] < thread:
                post_row_idxes.append(row_idx)
                post_col_idxes.append(col_idx)
        # row idxes is the idx of existed_bboxes, col_idxes is the idx of detected_bboxes
        # logger.debug(
        #     f"row_idxes length: {len(row_idxes)}, col_idxes length {len(col_idxes)}"
        # )
        return post_row_idxes, post_col_idxes

    def visualize_tracklets(self, ori_img: np.ndarray):
        for traklet in self.tracklets:
            box = traklet.location
            x0 = int(box.xmin)
            y0 = int(box.ymin)
            x1 = int(box.xmax)
            y1 = int(box.ymax)

            tracked_color = [0, 255, 0] # green 
            missed_color = [0, 0, 255] # red 
            logger.debug(f"Track id: {traklet.idx} status : {debug_str_map[traklet.status]}, tracked time: {traklet.track_time}, miss time: {traklet.miss_time}, lost time: {traklet.lost_time}")
            if traklet.status == TrackStatus.TRACKED:
                box_color = tracked_color
                show_str = f"id: {traklet.idx}, tk_time: {traklet.track_time}, tk_state: {debug_str_map[traklet.status]}"
            else:
                box_color = missed_color
                show_str = f"id: {traklet.idx}, miss_time: {traklet.miss_time}, tk_state: {debug_str_map[traklet.status]}"
            text_color = [255, 255, 255] # white
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(show_str, font, 0.4, 1)[0]

            cv2.rectangle(ori_img, (x0, y0), (x1, y1), box_color, 2)
            txt_bk_color = [int(0.5 * v) for v in box_color]
            cv2.rectangle(
                ori_img,
                (x0, y0+1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(ori_img, show_str, (x0, y0 + txt_size[1]), font, 0.4, text_color, thickness=1)
        return ori_img
