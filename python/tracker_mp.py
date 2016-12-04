from multiprocessing import Process
from multiprocessing import Queue
from kcftracker import KCFTracker

import numpy as np
import time


class TrackerMP(Process):
    def __init__(self, hog=True, fixed_window=False, multi_scale=True, input_queue=None, output_queue=None):
        Process.__init__(self)
        self.hog = hog
        self.fixed_window = fixed_window
        self.multi_scale = multi_scale
        self.tracker = KCFTracker(hog, fixed_window, multi_scale)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.image = None
        self.label = None
        self.is_valid = False

    def get_in_queue(self):
        return self.input_queue

    def get_out_queue(self):
        return self.output_queue

    def init(self, roi, image):
        return self.tracker.init(roi, image)
        # print('tracker init with image shape =', image.shape)

    def update(self, image):
        self.image = image
        # print('tracker update with image shape =', image.shape)

    def run(self):
        while True:
            # print('tracker ', os.getpid(), ' run with args', self.hog, self.fixed_window, self.multi_scale)
            input_dict = self.input_queue.get()
            cmd = input_dict.get('cmd')
            if cmd == 'init':
                roi = input_dict.get('roi')
                image = input_dict.get('image')
                self.label = input_dict.get('label')
                self.is_valid = self.tracker.init(roi, image)
            elif cmd == 'update':
                if self.is_valid:
                    image = input_dict.get('image')
                    roi, pv = self.tracker.update(image)
                    self.output_queue.put({'roi': roi, 'pv': pv, 'label': self.label})
                else:
                    self.output_queue.put(None)
            elif cmd == 'terminate':
                self.input_queue.close()
                self.output_queue.close()
                break


if __name__ == '__main__':

    task_queue = Queue()
    result_queue = Queue()

    width = 800
    height = 600
    offset_x = np.random.randint(0, width)
    offset_y = np.random.randint(0, height)

    # Start consumers
    num_consumers = 4
    print('Creating %d consumers' % num_consumers)
    all_trackers = [TrackerMP(True, False, True, task_queue, result_queue) for _ in range(num_consumers)]
    for i, tracker in enumerate(all_trackers):
        img = np.multiply(np.random.rand(height, width, 3), 255)
        tracker.init([offset_x, offset_y, width, height], img)
        tracker.start()

    for _ in range(10):
        t0 = time.time()
        for tracker in all_trackers:
            img = np.multiply(np.random.rand(height, width, 3), 255)
            task_queue.put(img)

        num_jobs = num_consumers
        while num_jobs > 0:
            ret = result_queue.get()
            num_jobs -= 1
            print(ret)
        t1 = time.time()
        print('exe time =', t1 - t0)

