import cv2

from img import Img


class Rect:
    x = None
    y = None
    w = None
    h = None

    def printit(self):
        print(str(self.x) + ',' + str(self.y) + ',' + str(self.w) + ',' + str(self.h))


class DragRectangle:
    keepWithin = Rect()
    outRect = Rect()
    anchor = Rect()
    sBlk = 4
    initialized = False
    image = None
    wname = ""
    returnflag = False
    active = False
    drag = False

    TL = False
    TM = False
    TR = False
    LM = False
    RM = False
    BL = False
    BM = False
    BR = False
    hold = False

    def __init__(self, Img, windowName, windowWidth, windowHeight):
        self.image = Img

        self.wname = windowName

        self.keepWithin.x = 0
        self.keepWithin.y = 0
        self.keepWithin.w = windowWidth
        self.keepWithin.h = windowHeight

        self.outRect.x = 0
        self.outRect.y = 0
        self.outRect.w = 0
        self.outRect.h = 0


class Drag:
    def __init__(self, bck: Img):
        self.dragObj = None
        self.bck = bck

    def dragrect(self, event, x, y, flags, _dragObj):
        self.dragObj = _dragObj
        if x < self.dragObj.keepWithin.x:
            x = self.dragObj.keepWithin.x
        if y < self.dragObj.keepWithin.y:
            y = self.dragObj.keepWithin.y
        if x > (self.dragObj.keepWithin.x + self.dragObj.keepWithin.w - 1):
            x = self.dragObj.keepWithin.x + self.dragObj.keepWithin.w - 1
        if y > (self.dragObj.keepWithin.y + self.dragObj.keepWithin.h - 1):
            y = self.dragObj.keepWithin.y + self.dragObj.keepWithin.h - 1

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseDown(x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.mouseUp()
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouseMove(x, y)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.mouseDoubleClick(x, y)

    @staticmethod
    def pointInRect(pX, pY, rX, rY, rW, rH):
        if rX <= pX <= (rX + rW) and rY <= pY <= (rY + rH):
            return True
        else:
            return False

    def mouseDoubleClick(self, eX, eY):
        if self.dragObj.active:
            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x, self.dragObj.outRect.y, self.dragObj.outRect.w,
                                self.dragObj.outRect.h):
                self.dragObj.returnflag = True
                cv2.destroyWindow(self.dragObj.wname)

    def mouseDown(self, eX, eY):
        if self.dragObj.active:
            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x - self.dragObj.sBlk,
                                self.dragObj.outRect.y - self.dragObj.sBlk,
                                self.dragObj.sBlk * 2, self.dragObj.sBlk * 2):
                self.dragObj.TL = True
                return
            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk,
                                self.dragObj.outRect.y - self.dragObj.sBlk,
                                self.dragObj.sBlk * 2, self.dragObj.sBlk * 2):
                self.dragObj.TR = True
                return
            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x - self.dragObj.sBlk,
                                self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk,
                                self.dragObj.sBlk * 2, self.dragObj.sBlk * 2):
                self.dragObj.BL = True
                return
            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk,
                                self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk,
                                self.dragObj.sBlk * 2, self.dragObj.sBlk * 2):
                self.dragObj.BR = True
                return

            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x + self.dragObj.outRect.w / 2 - self.dragObj.sBlk,
                                self.dragObj.outRect.y - self.dragObj.sBlk,
                                self.dragObj.sBlk * 2, self.dragObj.sBlk * 2):
                self.dragObj.TM = True
                return
            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x + self.dragObj.outRect.w / 2 - self.dragObj.sBlk,
                                self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk,
                                self.dragObj.sBlk * 2, self.dragObj.sBlk * 2):
                self.dragObj.BM = True
                return
            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x - self.dragObj.sBlk,
                                self.dragObj.outRect.y + self.dragObj.outRect.h / 2 - self.dragObj.sBlk,
                                self.dragObj.sBlk * 2, self.dragObj.sBlk * 2):
                self.dragObj.LM = True
                return
            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk,
                                self.dragObj.outRect.y + self.dragObj.outRect.h / 2 - self.dragObj.sBlk,
                                self.dragObj.sBlk * 2, self.dragObj.sBlk * 2):
                self.dragObj.RM = True
                return

            if Drag.pointInRect(eX, eY, self.dragObj.outRect.x, self.dragObj.outRect.y, self.dragObj.outRect.w,
                                self.dragObj.outRect.h):
                self.dragObj.anchor.x = eX - self.dragObj.outRect.x
                self.dragObj.anchor.w = self.dragObj.outRect.w - self.dragObj.anchor.x
                self.dragObj.anchor.y = eY - self.dragObj.outRect.y
                self.dragObj.anchor.h = self.dragObj.outRect.h - self.dragObj.anchor.y
                self.dragObj.hold = True

                return

        else:
            self.dragObj.outRect.x = eX
            self.dragObj.outRect.y = eY
            self.dragObj.drag = True
            self.dragObj.active = True
            return

    def mouseMove(self, eX, eY):
        if self.dragObj.drag & self.dragObj.active:
            self.dragObj.outRect.w = eX - self.dragObj.outRect.x
            self.dragObj.outRect.h = eY - self.dragObj.outRect.y
        elif self.dragObj.hold:
            self.dragObj.outRect.x = eX - self.dragObj.anchor.x
            self.dragObj.outRect.y = eY - self.dragObj.anchor.y

            if self.dragObj.outRect.x < self.dragObj.keepWithin.x:
                self.dragObj.outRect.x = self.dragObj.keepWithin.x
            if self.dragObj.outRect.y < self.dragObj.keepWithin.y:
                self.dragObj.outRect.y = self.dragObj.keepWithin.y
            if (self.dragObj.outRect.x + self.dragObj.outRect.w) > (
                    self.dragObj.keepWithin.x + self.dragObj.keepWithin.w - 1):
                self.dragObj.outRect.x = self.dragObj.keepWithin.x + self.dragObj.keepWithin.w - 1 - self.dragObj.outRect.w
            if (self.dragObj.outRect.y + self.dragObj.outRect.h) > (
                    self.dragObj.keepWithin.y + self.dragObj.keepWithin.h - 1):
                self.dragObj.outRect.y = self.dragObj.keepWithin.y + self.dragObj.keepWithin.h - 1 - self.dragObj.outRect.h
        elif self.dragObj.TL:
            self.dragObj.outRect.w = (self.dragObj.outRect.x + self.dragObj.outRect.w) - eX
            self.dragObj.outRect.h = (self.dragObj.outRect.y + self.dragObj.outRect.h) - eY
            self.dragObj.outRect.x = eX
            self.dragObj.outRect.y = eY
        elif self.dragObj.BR:
            self.dragObj.outRect.w = eX - self.dragObj.outRect.x
            self.dragObj.outRect.h = eY - self.dragObj.outRect.y
        elif self.dragObj.TR:
            self.dragObj.outRect.h = (self.dragObj.outRect.y + self.dragObj.outRect.h) - eY
            self.dragObj.outRect.y = eY
            self.dragObj.outRect.w = eX - self.dragObj.outRect.x
        elif self.dragObj.BL:
            self.dragObj.outRect.w = (self.dragObj.outRect.x + self.dragObj.outRect.w) - eX
            self.dragObj.outRect.x = eX
            self.dragObj.outRect.h = eY - self.dragObj.outRect.y

        elif self.dragObj.TM:
            self.dragObj.outRect.h = (self.dragObj.outRect.y + self.dragObj.outRect.h) - eY
            self.dragObj.outRect.y = eY
        elif self.dragObj.BM:
            self.dragObj.outRect.h = eY - self.dragObj.outRect.y
        elif self.dragObj.LM:
            self.dragObj.outRect.w = (self.dragObj.outRect.x + self.dragObj.outRect.w) - eX
            self.dragObj.outRect.x = eX
        elif self.dragObj.RM:
            self.dragObj.outRect.w = eX - self.dragObj.outRect.x
        self.clear_canvas_n_draw()

    def mouseUp(self):
        self.dragObj.drag = False
        self.disableResizeButtons()
        self.straightenUpRect()
        if self.dragObj.outRect.w == 0 or self.dragObj.outRect.h == 0:
            self.dragObj.active = False

        self.clear_canvas_n_draw()

    def disableResizeButtons(self):
        self.dragObj.TL = self.dragObj.TM = self.dragObj.TR = False
        self.dragObj.LM = self.dragObj.RM = False
        self.dragObj.BL = self.dragObj.BM = self.dragObj.BR = False
        self.dragObj.hold = False

    def straightenUpRect(self):
        """
        Make sure x, y, w, h of the Rect are positive
        """
        if self.dragObj.outRect.w < 0:
            self.dragObj.outRect.x = self.dragObj.outRect.x + self.dragObj.outRect.w
            self.dragObj.outRect.w = -self.dragObj.outRect.w
        if self.dragObj.outRect.h < 0:
            self.dragObj.outRect.y = self.dragObj.outRect.y + self.dragObj.outRect.h
            self.dragObj.outRect.h = -self.dragObj.outRect.h

    def clear_canvas_n_draw(self):
        tmp = self.dragObj.image.copy()
        cv2.rectangle(tmp, (self.dragObj.outRect.x, self.dragObj.outRect.y),
                      (self.dragObj.outRect.x + self.dragObj.outRect.w,
                       self.dragObj.outRect.y + self.dragObj.outRect.h), (0, 255, 0), 2)
        self.draw_select_markers(tmp)
        dst = cv2.addWeighted(tmp, 1, self.bck.image, .7, 0)
        cv2.imshow(self.dragObj.wname, dst)
        cv2.waitKey()

    def draw_select_markers(self, image):
        """
        Draw markers on the dragged rectangle
        """
        cv2.rectangle(image, (self.dragObj.outRect.x - self.dragObj.sBlk,
                              self.dragObj.outRect.y - self.dragObj.sBlk),
                      (self.dragObj.outRect.x - self.dragObj.sBlk + self.dragObj.sBlk * 2,
                       self.dragObj.outRect.y - self.dragObj.sBlk + self.dragObj.sBlk * 2),
                      (0, 255, 0), 2)
        cv2.rectangle(image, (self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk,
                              self.dragObj.outRect.y - self.dragObj.sBlk),
                      (self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk + self.dragObj.sBlk * 2,
                       self.dragObj.outRect.y - self.dragObj.sBlk + self.dragObj.sBlk * 2),
                      (0, 255, 0), 2)
        cv2.rectangle(image, (self.dragObj.outRect.x - self.dragObj.sBlk,
                              self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk),
                      (self.dragObj.outRect.x - self.dragObj.sBlk + self.dragObj.sBlk * 2,
                       self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk + self.dragObj.sBlk * 2),
                      (0, 255, 0), 2)
        cv2.rectangle(image, (self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk,
                              self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk),
                      (self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk + self.dragObj.sBlk * 2,
                       self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk + self.dragObj.sBlk * 2),
                      (0, 255, 0), 2)

        cv2.rectangle(image, (self.dragObj.outRect.x + int(self.dragObj.outRect.w / 2) - self.dragObj.sBlk,
                              self.dragObj.outRect.y - self.dragObj.sBlk),
                      (self.dragObj.outRect.x + int(
                          self.dragObj.outRect.w / 2) - self.dragObj.sBlk + self.dragObj.sBlk * 2,
                       self.dragObj.outRect.y - self.dragObj.sBlk + self.dragObj.sBlk * 2),
                      (0, 255, 0), 2)
        cv2.rectangle(image, (self.dragObj.outRect.x + int(self.dragObj.outRect.w / 2) - self.dragObj.sBlk,
                              self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk),
                      (self.dragObj.outRect.x + int(
                          self.dragObj.outRect.w / 2) - self.dragObj.sBlk + self.dragObj.sBlk * 2,
                       self.dragObj.outRect.y + self.dragObj.outRect.h - self.dragObj.sBlk + self.dragObj.sBlk * 2),
                      (0, 255, 0), 2)
        cv2.rectangle(image, (self.dragObj.outRect.x - self.dragObj.sBlk,
                              self.dragObj.outRect.y + int(self.dragObj.outRect.h / 2) - self.dragObj.sBlk),
                      (self.dragObj.outRect.x - self.dragObj.sBlk + self.dragObj.sBlk * 2,
                       self.dragObj.outRect.y + int(
                           self.dragObj.outRect.h / 2) - self.dragObj.sBlk + self.dragObj.sBlk * 2),
                      (0, 255, 0), 2)
        cv2.rectangle(image, (self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk,
                              self.dragObj.outRect.y + int(self.dragObj.outRect.h / 2) - self.dragObj.sBlk),
                      (self.dragObj.outRect.x + self.dragObj.outRect.w - self.dragObj.sBlk + self.dragObj.sBlk * 2,
                       self.dragObj.outRect.y + int(
                           self.dragObj.outRect.h / 2) - self.dragObj.sBlk + self.dragObj.sBlk * 2),
                      (0, 255, 0), 2)
