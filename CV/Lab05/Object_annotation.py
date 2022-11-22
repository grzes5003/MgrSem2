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
    # image *= 255

    rectI = dmr.DragRectangle(image, w_name, image_width, image_height)

    img = Img()

    img.image = 'img.png'

    drag = dmr.Drag(img)

    cv2.namedWindow(rectI.wname)
    cv2.imshow(rectI.wname, img.image)
    cv2.setMouseCallback(rectI.wname, drag.dragrect, rectI)

    def back(*args):
        pass
    cv2.createButton("Back", back, None, cv2.QT_PUSH_BUTTON, 1)

    while True:
        # cv2.imshow(w_name, np.concatenate((rectI.image, img.image), axis=0))
        # dst = cv2.addWeighted(rectI.image, .5, img.image, 1, 0)
        # cv2.imshow(w_name, dst)
        key = cv2.waitKey(1) & 0xFF
        print(f"key {key}")

        # KEYBOARD INTERACTIONS
        if key == ord('q'):
            cv2.destroyAllWindows()

        elif key == ord('s'):
            # save the image as such
            cv2.imwrite('mimi_colour.jpg', img)
            cv2.destroyAllWindows()

        elif key == ord('g'):
            # convert to grayscale and save it
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('mimi_gray.jpg', gray)
            cv2.destroyAllWindows()

        elif key == ord('t'):
            # write some text and save it
            text_image = cv2.putText(img, 'Miracles of OpenCV', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            cv2.imwrite('mimi_text.jpg', text_image)
            cv2.destroyAllWindows()

        if rectI.returnflag:
            break

    print("Dragged rectangle coordinates")
    print(str(rectI.outRect.x) + ',' + str(rectI.outRect.y) + ',' + \
          str(rectI.outRect.w) + ',' + str(rectI.outRect.h))

    # close all open windows
    cv2.destroyAllWindows()
