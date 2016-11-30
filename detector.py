from ssddetector import SSDDetector


class Detector:
    def __init__(self, pseudo=True, label_file=None, data_format=None):
        self.pseudo = pseudo
        self.detector = None
        if pseudo:
            self.frames = list()
            self.read_detection(label_file, data_format)
        else:
            self.detector = SSDDetector()

    def read_detection(self, label_file=None, data_format=None):
        self.frames = list()
        with open(label_file) as labels:
            for line in labels.readlines():
                self.frames.append(list())
                info = self.parse_label(line, data_format)
                if info.get('id') != -1:  # pass unknown objects
                    fi = info.get('frame')
                    self.frames[fi].append(info)

    def detect(self, frame=None, frame_num=0):
        if self.pseudo:
            ret = self.frames[frame_num]
        else:
            ret = self.detector.detect(frame)
        return ret

    @staticmethod
    def parse_label(label_line, data_format=''):
        if data_format == 'KITTI':
            val = label_line.split(' ')
            d = {'frame': int(val[0]), 'id': int(val[1]), 'object_class': val[2].strip('"'),
                 'x1': int(float(val[6])), 'y1': int(float(val[7])), 'x2': int(float(val[8])), 'y2': int(float(val[9]))}
        elif data_format == 'VTB':
            val = label_line.split('\t')
            d = {'frame': int(val[0]) - 1, 'id': int(val[1]), 'object_class': val[2].strip('"'),
                 'x1': int(float(val[3])), 'y1': int(float(val[4])),
                 'x2': int(float(val[3]) + float(val[5])), 'y2': int(float(val[4]) + float(val[6]))}
        else:
            val = label_line.split('\t')
            d = {'frame': int(val[0].lstrip('0')) - 1, 'id': int(val[1]), 'object_class': val[2].strip('"'),
                 'x1': int(float(val[3])), 'y1': int(float(val[4])), 'x2': int(float(val[5])), 'y2': int(float(val[6]))}
        return d
