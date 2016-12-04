

def iou_func(bbox1, bbox2):
    x1 = bbox1[0]
    y1 = bbox1[1]
    width1 = bbox1[2] - bbox1[0]
    height1 = bbox1[3] - bbox1[1]

    x2 = bbox2[0]
    y2 = bbox2[1]
    width2 = bbox2[2] - bbox2[0]
    height2 = bbox2[3] - bbox2[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    # return IOU
    return ratio


if __name__ == '__main__':
    print(iou((0, 0, 100, 100), (-10, 0, 90, 100)))
