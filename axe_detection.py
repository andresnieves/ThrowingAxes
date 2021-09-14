import cv2
import numpy as np


class AxeProcessor:
    time_to_score_sec = 2
    detect_background_frames = 60

    same_detection_threshhold = 100

    def __init__(self, fps, max_frames):
        self.max_frames = max_frames
        self.fps = fps

        self.axes_found = False
        self.background_frames = []
        self.background_avg = None
        self.current_frame = 0
        self.countdown_to_score = AxeProcessor.time_to_score_sec

        self.previous_detection = None

    def _find_axes(self, frame):
        output = frame.copy()
        dframe = cv2.absdiff(frame, self.background_avg)
        blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        ret, tframe = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        (cnts, _) = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, cnts, -1, (0, 0, 255), 1)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(output, [box], 0, (0, 191, 255), 1)
        return output, cnts

    def _calculate_background(self):
        self.background_avg = np.average(self.background_frames, axis=0).astype(dtype=np.uint8)

    def _calculate_distances(self):
        pass

    def process_frame(self, frame):
        self.current_frame += 1
        if self.current_frame < AxeProcessor.detect_background_frames:
            # Collect background frames
            self.background_frames.append(frame)
            return None, []

        if self.current_frame == AxeProcessor.detect_background_frames:
            # Calculate background average
            self._calculate_background()

        return self._find_axes(frame)
