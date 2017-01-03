from util import *
from math import pow, sqrt


def xy_center(obj):
    x_center = (obj.get('x1') + obj.get('x2')) / 2
    y_center = (obj.get('y1') + obj.get('y2')) / 2
    return x_center, y_center


def distance(pos1, pos2):
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def find_nearest(obj, obj_list):
    oc = xy_center(obj)
    nearest_dist = 10000
    nearest_obj = None
    for t in obj_list:
        tc = xy_center(t)
        dist = distance(oc, tc)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_obj = t
    return nearest_obj


def get_bbox(obj):
    return obj.get('x1'), obj.get('y1'), obj.get('x2'), obj.get('y2')


def parse_label(label_line, label_format):
    if label_format == 'kitti':
        val = label_line.split(' ')
        d = {'frame': int(val[0]), 'id': int(val[1]), 'object_class': val[2].strip('"'),
             'x1': int(float(val[6])), 'y1': int(float(val[7])), 'x2': int(float(val[8])), 'y2': int(float(val[9]))}
    elif label_format == 'fdt':
        val = label_line.split(',')
        d = {'frame': int(val[0]), 'x1': int(float(val[1])), 'y1': int(float(val[2])),
             'x2': int(float(val[3])), 'y2': int(float(val[4]))}
    return d


def get_detections(kitti_det_file, label_format):
    frames = list()
    with open(kitti_det_file) as labels:
        for line in labels.readlines():
            info = parse_label(line, label_format)
            if info.get('id') != -1:  # pass unknown objects
                fi = info.get('frame')
                while len(frames) <= fi:
                    frames.append(list())  # create empty list to that frame
                frames[fi].append(info)
    return frames


def evaluate_accuracy(test_frames, golden_frames):
    sum_iou = 0.0
    for i in range(min(len(test_frames), len(golden_frames))):
        for g in golden_frames[i]:
            nearest_obj = find_nearest(g, test_frames[i])
            if nearest_obj is None:
                continue
            iou = iou_func(get_bbox(g), get_bbox(nearest_obj))
            sum_iou += iou
            # if iou != 0:
                # print('g', g)
                # print('nearest_obj', nearest_obj)
                # print('iou =', iou)
    return sum_iou


def evaluation_main():
    fdt_result = get_detections('bbox.txt', 'fdt')
    golden = get_detections('/home/cory/cedl/KITTI/labels/0001.txt', 'kitti')

    golden_iou_sum = evaluate_accuracy(golden, golden)
    test_iou_sum = evaluate_accuracy(fdt_result, golden)

    print('golden_iou_sum = {:.2f}'.format(golden_iou_sum))
    print('test_iou_sum = {:.2f}'.format(test_iou_sum))


if __name__ == '__main__':
    evaluation_main()