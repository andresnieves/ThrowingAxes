import cv2
import numpy as np

from src.utils import warp_board


def detect_circles(frame_gray, original, cnt_circles=1):
    '''
    param_1 = 200: Upper threshold for the internal Canny edge detector.
    param_2 = 100*: Threshold for center detection.
    min_radius = 0: Minimum radius to be detected. If unknown, put zero as default.
    max_radius = 0: Maximum radius to be detected. If unknown, put zero as default.
    '''
    circles = cv2.HoughCircles(frame_gray,
                               cv2.HOUGH_GRADIENT,
                               1,
                               20,
                               param1=170,
                               param2=150,
                               minRadius=30,
                               maxRadius=0)

    circles = np.uint16(np.around(circles))
    circles_found = []
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(original, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(original, (i[0], i[1]), 2, (0, 0, 255), 3)
        rad = int(i[2] / cnt_circles)
        current_rad = rad
        for j in range(1, cnt_circles):
            cv2.circle(original, (i[0], i[1]), current_rad, (0, 200, 0), 1)
            current_rad += rad
        circles_found.append(((i[0], i[1]), i[2]))
    return original, circles_found


def detect_board(frame, cnt_circles):
    """
    Steps to detect board:
    1) warp image
    2) gray
    2) blur
    3) dilate
    4) thresholding
    4) find contours
    5) find hough circles
    6) calculate other radius

    :param frame:
    :return:
    """
    original = frame  # warp_board(frame)
    frame = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, 150, 255)
    frame, circles = detect_circles(frame, original, cnt_circles)
    return frame, circles


img = cv2.imread('../images/white.png')
#img = warp_board(img)

frame, _ = detect_board(img, 5)

cv2.imshow("test", frame)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
