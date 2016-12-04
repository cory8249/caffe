from __future__ import print_function

import cv2
import sys
from time import time
import os

from multiprocessing import Queue
from tracker_mp import TrackerMP
from config import *
from util import *
from detector import Detector


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
    duration_smooth = 0.01
    sum_pv = 0.0
    sum_iou = 0.0
    no_result_count = 0
    id_generator = Generator()
    default_tracker_life = 50
    iou_kill_threshold = 0.6

    # ============  main tracking loop  ============ #
    for current_frame in range(frames_count):
        t0 = time()
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

        print('frame %d: ' % current_frame)

        # select mode by current frame count (periodic prediction)
        init_tracking = (current_frame % detection_period == 0)

        if init_tracking:
            # run detection
            detections = detector.detect(frame, current_frame)
            print(detections)

            detections_sorted = sorted(detections, key=lambda d: d.get('id'))
            print(' ---------------- # trackers = %d' % len(detections_sorted), end='')
            for dt in detections_sorted:
                x1 = dt.get('x1')
                y1 = dt.get('y1')
                w = dt.get('x2') - x1
                h = dt.get('y2') - y1
                label = dt.get('label')
                tid = id_generator.get_next_id()

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
                    print('%d kill itself as life end' % tid)
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
                        print('iou(%d,%d)=%.2f' % (tid, nid, iou))
                    if iou > iou_kill_threshold:
                        print('%d killed by %d with iou = %.2f' % (tid, nid, iou))
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
                    print('ret == None, tid = %d' % tid)
                    no_result_count += 1
                    continue
                roi = map(int, ret.get('roi'))
                bbox = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
                all_trackers[tid].update({'x1': roi[0], 'y1': roi[1], 'w': roi[2], 'h': roi[3]})
                pv = ret.get('pv')
                pv_threshold = 0.25
                active = pv > pv_threshold
                sum_pv += pv
                label = ret.get('label')

                if imshow_enable or imwrite_enable:
                    if active:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                      (0, 255, 255), 1)
                        cv2.putText(frame, '%.2f' % pv, (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 1)
                        cv2.putText(frame, '%d-%s' % (tid, label), (bbox[0], bbox[1] + roi[3] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 1)
                    else:
                        cv2.putText(frame, '%.2f' % pv, (bbox[0], bbox[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 0, 0), 1)

        if imshow_enable or imwrite_enable:
            t1 = time()
            duration_smooth = 0.8 * duration_smooth + 0.2 * (t1 - t0)
            fps = 1 / duration_smooth
            print(' fsp = %4f' % fps, end='')
            cv2.putText(frame, 'FPS: %.2f' % fps, (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.imshow('tracking', frame)
            c = cv2.waitKey(1) & 0xFF
            if c == 27 or c == ord('q'):
                break

        if imwrite_enable:
            cv2.imwrite('../output/frame_%06d.jpg' % current_frame, frame)

        print(' sum_pv = %f' % sum_pv)

    # terminate all trackers after all frames are processed
    for tid, tracker in all_trackers.items():
        terminate_tracker(tracker)

    if input_mode == 'video':
        cap.release()
    if imshow_enable:
        cv2.destroyAllWindows()

    print('no_result_count = %d' % no_result_count)
    print('sum_iou = %f' % sum_iou)


# ============   Usage: run_fdt.py <filename> <det_result>   ============ #

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append(default_input_path)
        sys.argv.append(default_det_path)
        sys.argv.append(default_data_format)
    assert len(sys.argv) == 4
    if imwrite_enable and not os.path.exists('output'):
        os.mkdir('output')

    fdt_main(input_path=sys.argv[1], label_file=sys.argv[2], data_format=sys.argv[3])
