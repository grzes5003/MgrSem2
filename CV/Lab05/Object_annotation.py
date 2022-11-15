import sys

import cv2
import numpy as np

from img import Img

sys.setrecursionlimit(10 ** 9)

import drag_and_move_rectangle as dmr


def app():
    w_name = "Annotation for Object Detection"
    img = Img()
    img.image = 'img.png'

    image_height, image_width, image_depth = img.shape
    # image_width = 320
    # image_height = 240
    image = np.zeros([image_height, image_width, 4], dtype=np.uint8)
    image *= 255

    rectI = dmr.DragRectangle(image, w_name, image_width, image_height)

    img = Img()

    img.image = 'img.png'

    drag = dmr.Drag()

    cv2.namedWindow(rectI.wname)
    cv2.imshow(rectI.wname, img.image)
    cv2.setMouseCallback(rectI.wname, lambda a, b, c, d, e: drag.dragrect(a, b, c, d), rectI)

    while True:
        # cv2.imshow(w_name, np.concatenate((rectI.image, img.image), axis=0))
        dst = cv2.addWeighted(rectI.image, .5, img.image, 1, 0)
        cv2.imshow(w_name, dst)
        key = cv2.waitKey(1) & 0xFF

        if rectI.returnflag:
            break

    print("Dragged rectangle coordinates")
    print(str(rectI.outRect.x) + ',' + str(rectI.outRect.y) + ',' + \
          str(rectI.outRect.w) + ',' + str(rectI.outRect.h))

    # close all open windows
    cv2.destroyAllWindows()
