import cv2
import numpy as np
from scipy.spatial import distance as dist
import math


def warp_board(frame):
    rows, cols, ch = frame.shape
    pts1 = np.float32(
        [[0, rows],
         [cols, rows],
         [0, 0],
         [cols, 0]]
    )
    pts2 = np.float32(
        [[0, rows],
         [cols * 1.05, rows * 0.949],
         [cols * 0.1, rows * 0.1],
         [cols, 0]]
    )
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(frame, M, (cols, rows))
    return dst


def dist_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def resize_frame(frame):
    return cv2.resize(frame, (640, 480))


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def top_most_contour_point(cnt):
    top = [0, 480]
    for c in cnt:
        if c[0][1] < top[1]:
            top = c[0]
    return top[0], top[1]


def average_points_contour(cnt):
    sum = 0
    for c in cnt:
        sum += c[0][1]
    return int(sum / len(cnt))


def filter_topmost_contour(cnt, base):
    return [x[0][1] < base for x in cnt]


def score_point(x, y, x0, y0, r):
    rad = int(r / 5)
    current_rad = rad
    scores = []
    for j in range(1, 5):
        scores.append((current_rad, 6-j))
        current_rad += rad
    for rad, score in scores:
        if dist_points((x, y), (x0, y0)) <= rad:
            return score
    return 0
