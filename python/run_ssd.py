# coding: utf-8

# Detection with SSD

from __future__ import print_function

from ssddetector import *
from config import *


def ssd_test_image(image_path=None):
    frame = cv2.imread(image_path)  # BGR
    assert frame is not None

    ssd_detector = SSDDetector()
    det = ssd_detector.detect(frame=frame, conf_threshold=0.6)
    print(det)
    ssd_detector.plot(frame, det)
    if imshow_enable:
        cv2.waitKey(0)


def ssd_test_video(video_path=None):
    # Open video file as cap
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError

    ssd_detector = SSDDetector()
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        det = ssd_detector.detect(frame=frame, conf_threshold=0.6)
        print(det)
        ssd_detector.plot(frame, det)
        frame_i += 1


if __name__ == '__main__':
    # I/O Path
    if len(sys.argv) == 1:
        sys.argv.append(default_input_path)

    if len(sys.argv) < 2:
        print('run_ssd.py <input_path>')
        sys.exit(1)

    input_path = sys.argv[1]
    if input_path.find('.mp4') != -1:
        ssd_test_video(video_path=input_path)
    else:
        ssd_test_image(image_path=input_path)

