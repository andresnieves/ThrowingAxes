import cv2

from src.axe_detection import AxeProcessor
from src.board_detection import detect_board
from src.utils import resize_frame


def load_video_stream(origin):
    video_stream = cv2.VideoCapture(origin)
    frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    return video_stream, fps, frames


def process_frame(original_frame):
    frame = original_frame.copy()
    frame = resize_frame(frame)
    board, _ = detect_board(frame, 4)
    axes_output, axes = axe_processor.process_frame(frame)

    return board, axes_output


def process_video_stream(stream):
    # Check if camera opened successfully
    if not stream.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    current_frame = 0
    while stream.isOpened():
        # Capture frame-by-frame
        ret, frame = stream.read()
        if ret:

            # Display the resulting frame
            board, axes = process_frame(frame)

            cv2.imshow('Board', board)
            if axes is not None:
                cv2.imshow('Axes', axes)

            current_frame += 1

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    stream.release()

    # Closes all the frames
    cv2.destroyAllWindows()


stream, fps, frames_cnt = load_video_stream('../video/up_axes_short.mp4')
axe_processor = AxeProcessor(fps, frames_cnt)
process_video_stream(stream)

