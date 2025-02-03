import numpy as np
from typing import List


class Bbox2d(object):
    """
        surpose to be top-left & bottom-right
    """
    EPS=1e-6
    def __init__(self, 
                 xmin: float, 
                 ymin: float,
                 xmax: float,
                 ymax: float) -> None:
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.w = self.xmax - self.xmin 
        self.h = self.ymax - self.ymin
        self.area = self.w * self.h

    def _iou(self, box2: "Bbox2d"):
        # intersection 
        xmin = max(self.xmin, box2.xmin)
        ymin = max(self.ymin, box2.ymin)
        xmax = min(self.xmax, box2.xmax)
        ymax = min(self.ymax, box2.ymax)

        inter_area = abs(max(xmax-xmin, 0) * max(ymax-ymin, 0))

        # union
        union_area = self.area + box2.area - inter_area
        iou = inter_area / (union_area + self.EPS)
        return iou

    def __str__(self) -> str:
        return f"xmin: {self.xmin}, \n ymin: {self.ymin}, \nxmax: {self.xmax}, \nymax: {self.ymax}"

def get_cost_matrix(bboxes1: List[Bbox2d],
                    bboxes2: List[Bbox2d]):
    cost_matrix = np.zeros((len(bboxes1), len(bboxes2)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    
    for i in range(len(bboxes1)):
        for j in range(len(bboxes2)):
            cost_matrix[i][j] = 1-bboxes1[i]._iou(bboxes2[j])
    return cost_matrix

if __name__ == "__main__":
    box1 = Bbox2d(0, 0, 10, 10)
    box2 = Bbox2d(1, 1, 11, 11)
    box3 = Bbox2d(12, 12, 22, 22)
    box4 = Bbox2d(0, 0, 2, 2)
    box5 = Bbox2d(1, 1, 3, 3)

    print(box1._iou(box2)) # 0.6806722631876281
    print(box2._iou(box1))
    print(box1._iou(box3)) # 0.
    print(box4._iou(box5)) # 0.1428571224489825
    print(box4._iou(box1))

    print(get_cost_matrix([box2, box4, box5], [box1, box3]))
    import pdb; pdb.set_trace()

