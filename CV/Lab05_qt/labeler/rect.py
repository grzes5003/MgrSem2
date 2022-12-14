from PySide6 import QtWidgets, QtCore, QtGui


class Rect(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(30, 30, 600, 400)
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.show()

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        br = QtGui.QBrush(QtGui.QColor(100, 10, 10, 40))
        qp.setBrush(br)
        qp.drawRect(QtCore.QRect(self.begin, self.end))

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        # self.update()

    def get_points(self) -> str:
        x, y = self.begin.toTuple()
        width, height = x - self.end.x(), y - self.end.y()
        return f'{x} {y} {width} {height}'
