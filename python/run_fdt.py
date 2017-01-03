from __future__ import print_function
import argparse
import cv2
import numpy as np
import sys
import math
from time import time
import os
import logging

from multiprocessing import Queue
from tracker_mp import TrackerMP
from config import *
from util import *
from detector import Detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=default_input_path,
                        help="input path of video")
    parser.add_argument("--label_file", type=str, default=default_det_path,
                        help="input path of detection label file")
    parser.add_argument("--data_format", type=str, default=default_data_format,
                        help="input data format")
    return parser.parse_args()


def get_logger():
    # create logger with 'spam_application'
    logger = logging.getLogger('fdt')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('fdt.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class Generator:
    def __init__(self):
        self.id = 0

    def get_next_id(self):
        new_id = self.id
        self.id += 1
        return new_id

    def current_id(self):
        return self.id


def terminate_tracker(tracker):
    tk_process = tracker['process']
    tk_process.get_in_queue().put({'cmd': 'terminate'})
    if tk_process.is_alive():
        tk_process.terminate()


def add_padding(frame, padding_size):
    npad = ((padding_size, padding_size), (0, 0), (0, 0))
    ret = np.pad(frame, pad_width=npad, mode='constant', constant_values=0)
    ret = ret.astype(np.uint8)
    return ret


def remove_padding(frame, size):
    ret = frame[size:-size, :, :]
    return ret


bbox_file = open('bbox.txt', 'w')


def dump_roi_to_file(frame_id, roi):
    bbox_file.write('%d,%d,%d,%d,%d\n' % (frame_id, roi[0], roi[1], roi[2], roi[3]))


def fdt_main(input_path=None, label_file=None, data_format=None):

    if input_path.find('mp4') != -1:
        input_mode = 'video'
    else:
        input_mode = 'image'

    # object detector
    detector = Detector(pseudo=False, label_file=label_file, data_format=data_format)

    if input_mode == 'video':
        cap = cv2.VideoCapture(input_path)
        assert cap.isOpened()
        frames_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        files_in_dir = sorted([f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))])
        frames_count = len(files_in_dir)

    all_trackers = dict()
    sum_pv = 0.0
    no_result_count = 0
    id_generator = Generator()
    default_tracker_life = 30
    iou_kill_threshold = 0.6
    pv_threshold = 0.25
    logger = get_logger()

    # ============  main tracking loop  ============ #
    loop_begin = time()
    for current_frame in range(frames_count):
        begin_time = time()
        if input_mode == 'video':
            ret, frame = cap.read()
        else:
            current_frame_path = input_path + '/%06d.jpg' % (current_frame + 1)
            frame = cv2.imread(current_frame_path)

        if frame is None:
            print('read image/video error at frame', current_frame)
            if input_mode == 'image':
                print(current_frame_path)
            raise IOError

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        padding_size = (frame_width - frame_height) / 2
        logger.info('frame %d: ' % current_frame)

        # select mode by current frame count (periodic prediction)
        init_tracking = (current_frame % detection_period == 0)

        if init_tracking:

            if padding_enable:
                frame = add_padding(frame, padding_size)

            # run detection
            t1 = time()
            detections = detector.detect(frame, current_frame)
            t2 = time()

            if padding_enable:
                frame = remove_padding(frame, padding_size)

            logger.debug(detections)
            logger.info('detection time = %0.4f' % (t2 - t1))

            detections_sorted = sorted(detections, key=lambda d: d.get('id'))
            logger.debug(' ---------------- # trackers = %d' % len(detections_sorted))
            for dt in detections_sorted:
                x1 = dt.get('x1')
                y1 = dt.get('y1')
                x2 = dt.get('x2')
                y2 = dt.get('y2')
                if padding_enable:
                    y1 -= padding_size
                    y2 -= padding_size
                dump_roi_to_file(current_frame,  (x1, y1, x2, y2))
                w = x2 - x1
                h = y2 - y1
                label = dt.get('label')
                conf = dt.get('conf')
                tid = id_generator.get_next_id()
                roi = map(int, [x1, y1, w, h])
                bbox = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0, 255, 0), 2)
                cv2.putText(frame, '%.2f' % conf, (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 1)
                cv2.putText(frame, '%d-%s' % (tid, label), (bbox[0], bbox[1] + roi[3] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 1)

                if detection_period != 1:
                    tk_process = TrackerMP(hog=True, fixed_window=False, multi_scale=True,
                                           input_queue=Queue(), output_queue=Queue())
                    tk_process.start()
                    tk_process.get_in_queue().put({'cmd': 'init',
                                                   'label': label,
                                                   'roi': [x1, y1, w, h],
                                                   'image': frame})
                    all_trackers[tid] = {'process': tk_process, 'life': default_tracker_life,
                                         'x1': x1, 'y1': y1, 'w': w, 'h': h, 'label': label}  # add to trackers' dict

        else:

            # check old trackers
            # * every tracker life - 1
            # * kill tracker if life == 0
            # * kill old tracker if IOU between new trackers > threshold
            sorted_trackers = sorted(all_trackers.items())
            for tid, tracker in sorted_trackers:
                life = tracker['life']
                if life > 0:
                    tracker['life'] = life - 1
                else:
                    logger.debug('%d kill itself as life end' % tid)
                    terminate_tracker(tracker)
                    del all_trackers[tid]
                    continue

                roi1 = (tracker['x1'], tracker['y1'], tracker['x1'] + tracker['w'], tracker['y1'] + tracker['h'])
                for nid in range(tid + 1, id_generator.current_id()):
                    tnx = all_trackers.get(nid)
                    if tnx is None:
                        break
                    roi2 = (tnx['x1'], tnx['y1'], tnx['x1'] + tnx['w'], tnx['y1'] + tnx['h'])
                    iou = iou_func(roi1, roi2)
                    if iou > 0:
                        logger.debug('iou(%d,%d)=%.2f' % (tid, nid, iou))
                    if iou > iou_kill_threshold:
                        logger.debug('%d killed by %d with iou = %.2f' % (tid, nid, iou))
                        terminate_tracker(tracker)
                        del all_trackers[tid]
                        break

            for tid, tracker in all_trackers.items():
                tk_process = tracker['process']
                tk_process.get_in_queue().put({'cmd': 'update', 'image': frame})

            # trackers  will calculate in their sub-processes

            for tid, tracker in all_trackers.items():
                tk_process = tracker['process']
                ret = tk_process.get_out_queue().get()
                if ret is None:
                    # something wrong with this tracker, pass it
                    logger.debug('ret == None, tid = %d' % tid)
                    no_result_count += 1
                    continue
                roi = map(int, ret.get('roi'))
                bbox = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
                all_trackers[tid].update({'x1': roi[0], 'y1': roi[1], 'w': roi[2], 'h': roi[3]})
                pv = ret.get('pv')
                logger.debug('tid %d, pv = %.2f' % (tid, pv))
                active = pv > pv_threshold
                sum_pv += pv
                label = ret.get('label')

                life = tracker['life']
                logger.debug('tid %d, life = %d' % (tid, life))
                color_decay = max(math.pow((life + 1.0) / default_tracker_life, 1.2), 0.4)
                bbox_color = (0, 255 * color_decay, 0)

                if imshow_enable or imwrite_enable:
                    if active:
                        dump_roi_to_file(current_frame, bbox)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                      bbox_color, 2)
                        cv2.putText(frame, '%.2f' % pv, (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 1)
                        cv2.putText(frame, '%d-%s' % (tid, label), (bbox[0], bbox[1] + roi[3] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 1)
                    else:
                        pass
                        #cv2.putText(frame, '%.2f' % pv, (bbox[0], bbox[1]),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        #            (255, 0, 0), 1)

        end_time = time()
        fps = 1 / (end_time - begin_time)
        logger.info('fsp = %4f' % fps)

        if imshow_enable or imwrite_enable:
            cv2.putText(frame, 'FPS: %.2f' % fps, (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        if imshow_enable:
            cv2.imshow('tracking', frame)
            c = cv2.waitKey(1) & 0xFF
            if c == 27 or c == ord('q'):
                break
        if imwrite_enable:
            cv2.imwrite(default_output_path + '/frame_%06d.jpg' % current_frame, frame)

        logger.debug('sum_pv = %f' % sum_pv)

    # terminate all trackers after all frames are processed
    for tid, tracker in all_trackers.items():
        terminate_tracker(tracker)

    if input_mode == 'video':
        cap.release()
    if imshow_enable:
        cv2.destroyAllWindows()

    total_time = time() - loop_begin
    avg_fps = frames_count / total_time
    print('total time = {:.4f} second(s)'.format(total_time))
    print('avg fsp = {:.4f} fps'.format(avg_fps))


# ============   Usage: run_fdt.py <filename> <det_result>   ============ #

if __name__ == '__main__':
    args = parse_args()

    if imwrite_enable and not os.path.exists(default_output_path):
        os.mkdir(default_output_path)

    fdt_main(input_path=args.input, label_file=args.label_file, data_format=args.data_format)
