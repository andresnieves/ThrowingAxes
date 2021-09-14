import cv2
import numpy as np

from src.board_detection import warp_board, detect_board
from src.utils import order_points, average_points_contour, filter_topmost_contour, top_most_contour_point, dist_points, \
    score_point

video_stream = cv2.VideoCapture('../video/up_axes.mp4')
_, circles = detect_board(warp_board(cv2.imread('../output/frame100.jpg')), 5)

thresh = 90
thresh_max = 255
blur_cluster = 11
contrast_v = 25  # 1.25
brightness = 10
min_rectangle_length = 12

current_frame = 0
background_frames = []


def normalize(f):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    contrast = float(f'1.{contrast_v}')
    f[:, :, 2] = np.clip(contrast * f[:, :, 2] + brightness, 0, 255)
    f = cv2.cvtColor(f, cv2.COLOR_HSV2BGR)
    return f


def add_score_text(frame, x, score):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (x - 50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    return cv2.putText(frame, f'Score {score}!', org, font,
                       fontScale, color, thickness, cv2.LINE_AA)


def process_frame(frame):
    output = frame.copy()
    output = normalize(output)
    dframe = cv2.absdiff(output, normalize(background_avg))
    blurred = cv2.GaussianBlur(dframe, (blur_cluster, blur_cluster), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret, tframe = cv2.threshold(gray, thresh, thresh_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) < 10:
        for cnt in cnts:
            # x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            tl, _, _, bl = order_points(box)
            if dist_points(tl, bl) >= min_rectangle_length:
                cv2.drawContours(output, [box], 0, (0, 191, 255), 1)
                avg_cnt = average_points_contour(cnt)
                filtered_cnt = cnt[filter_topmost_contour(cnt, avg_cnt)]
                top_x, top_y = top_most_contour_point(filtered_cnt)
                for ((x0, y0), r) in circles:
                    score = score_point(top_x, top_y, x0, y0, r)
                    if score > 0:
                        add_score_text(output, x0, score)
                cv2.drawContours(output, [filtered_cnt], -1, (0, 0, 255), 1)
    for ((x, y), r) in circles:
        # draw the outer circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        rad = int(r / 5)
        current_rad = rad
        for j in range(1, 5):
            cv2.circle(output, (x, y), current_rad, (0, 200, 0), 1)
            current_rad += rad

    return output


if not video_stream.isOpened():
    print("Error opening video stream or file")

writer = cv2.VideoWriter("process_output.mp4",
                         cv2.VideoWriter_fourcc(*"MP4V"), 30, (640, 480))

# Read until video is completed
while video_stream.isOpened():
    current_frame += 1
    print(f"Frame {current_frame} of {video_stream.get(cv2.CAP_PROP_FRAME_COUNT)}")
    ret, frame = video_stream.read()
    if ret:
        transformed = warp_board(cv2.resize(frame, (640, 480)))
        if len(background_frames) < 60:
            background_frames.append(transformed)
        if len(background_frames) == 60:
            background_avg = np.average(background_frames, axis=0).astype(dtype=np.uint8)
        if current_frame > 60:
            f = process_frame(transformed)
            for i in range(2):
                writer.write(cv2.resize(f, (640, 480)))
    else:
        break

# When everything done, release the video capture object
print("[DONE]")
video_stream.release()
writer.release()
