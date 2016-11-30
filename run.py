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


if __name__ == '__main__':

    # ============   Usage: run.py <filename> <det_result>   ============ #

    if len(sys.argv) == 1:
        sys.argv.append(default_input_path)
        sys.argv.append(default_det_path)
        sys.argv.append(default_data_format)
    assert len(sys.argv) == 4
    if not os.path.exists('output'):
        os.mkdir('output')

    input_v_path = sys.argv[1]
    label_file = sys.argv[2]
    data_format = sys.argv[3]
    if input_v_path.find('mp4') != -1:
        input_mode = 'video'
    else:
        input_mode = 'image'

    # object detector
    detector = Detector(pseudo=False, label_file=label_file, data_format=data_format)

    if input_mode == 'video':
        cap = cv2.VideoCapture(input_v_path)
        assert cap.isOpened()
        frames_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        files_in_dir = sorted([f for f in os.listdir(input_v_path) if os.path.isfile(os.path.join(input_v_path, f))])
        frames_count = len(files_in_dir)

    all_trackers = dict()
    tracker_valid = dict()
    duration = 0.01
    duration_smooth = 0.01
    sum_pv = 0.0
    sum_iou = 0.0
    no_result_count = 0

    # ============  main tracking loop  ============ #
    for current_frame in range(frames_count):
        if input_mode == 'video':
            ret, frame = cap.read()
        else:
            current_frame_path = input_v_path + '/%06d.jpg' % (current_frame + 1)
            frame = cv2.imread(current_frame_path)

        if frame is None:
            print('read image/video error at frame', current_frame)
            if input_mode == 'image':
                print(current_frame_path)
            raise IOError

        print('frame %d' % current_frame, end='')

        # select mode by current frame count (periodic prediction)
        initTracking = (current_frame % detection_period == 0)

        if initTracking:
            # run detection
            det = detector.detect(frame, current_frame)
            print(det)

            # invalidate old trackers
            for tid, tracker in all_trackers.items():
                tracker_valid.update({tid: False})

            all_targets = sorted(det, key=lambda d: d.get('id'))
            print(' ---------------- # trackers = %d' % len(all_targets), end='')
            for target in all_targets:
                # print(target)
                ix = target.get('x1')
                iy = target.get('y1')
                w = target.get('x2') - ix
                h = target.get('y2') - iy
                tid = target.get('id')
                label = target.get('label')

                tracker = all_trackers.get(tid)
                if tracker is None:
                    tracker = TrackerMP(hog=True, fixed_window=False, multi_scale=True,
                                        input_queue=Queue(), output_queue=Queue())
                    tracker.start()
                tracker.get_in_queue().put({'cmd': 'init',
                                            'label': label,
                                            'roi': [ix, iy, w, h],
                                            'image': frame})
                all_trackers.update({tid: tracker})  # add to trackers' dict
                tracker_valid.update({tid: True})

            initTracking = False
            onTracking = True

        elif onTracking:
            t0 = time()
            for tracker in [v for (k, v) in all_trackers.items() if tracker_valid.get(k)]:
                tracker.get_in_queue().put({'cmd': 'update',
                                            'image': frame})
            # trackers  will calculate in their sub-processes
            if False:
                det = detector.detect(frame, current_frame)
                print(det)
                ground_truth = sorted(det, key=lambda d: d.get('id'))
                ground_truth_id = [x.get('id') for x in ground_truth]
            for (tid, tracker) in [(k, v) for (k, v) in all_trackers.items() if tracker_valid.get(k)]:
                ret = tracker.get_out_queue().get()
                if ret is None:
                    # something wrong with this tracker, pass it
                    print('ret == None, tid = %d' % tid)
                    no_result_count += 1
                    continue
                roi = map(int, ret.get('roi'))
                bbox = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
                pv = ret.get('pv')
                pv_threshold = 0.25
                active = pv > pv_threshold
                sum_pv += pv
                label = ret.get('label')

                if False and active and tid in ground_truth_id:
                    # print(target)
                    ix = ground_truth_id.index(tid)
                    gt = ground_truth[ix]
                    gt_bbox = (gt.get('x1'), gt.get('y1'), gt.get('x2'), gt.get('y2'))
                    iou = iou_func(bbox, gt_bbox)
                    sum_iou += iou
                    # print('%d iou = %f' % (tid, iou))

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
            t1 = time()
            duration = t1 - t0
            duration_smooth = 0.8 * duration_smooth + 0.2 * (t1 - t0)
            fps = 1 / duration_smooth
            print(' fsp = %4f' % fps, end='')
            if imshow_enable or imwrite_enable:
                cv2.putText(frame, 'FPS: ' + str(fps)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

        if imshow_enable:
            cv2.imshow('tracking', frame)
            c = cv2.waitKey(1) & 0xFF
            if c == 27 or c == ord('q'):
                break

        if imwrite_enable:
            cv2.imwrite('output/frame_%06d.jpg' % current_frame, frame)

        print(' sum_pv = %f' % sum_pv)

    # terminate all trackers after all frames are processed
    for tid, tracker in all_trackers.items():
        tracker.get_in_queue().put({'cmd': 'terminate'})
        if tracker.is_alive():
            tracker.terminate()

    if input_mode == 'video':
        cap.release()
    if imshow_enable:
        cv2.destroyAllWindows()

    print('no_result_count = %d' % no_result_count)
    print('sum_iou = %f' % sum_iou)
