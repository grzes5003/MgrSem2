# This Python file uses the following encoding: utf-8
import sys

from PySide6 import QtWidgets
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QFileDialog

from image_view import ImageView
#from PyQt5 import QLabel, QPixmap
#from PyQt5.QtGui import QPixmap

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        self.ui.newButton.clicked.connect(self.load_image)

#        newAct = QAction('New', self)
#        self.numpyPicture = QPixmap('./img.png')
#        newAct.triggered.connect(self.new_pic)
#        toolbar = self.addToolBar('Exit')
#        toolbar.addAction(newAct)
#        self.setGeometry(300, 300, 350, 250)

#        self.im = QPixmap('./img.png')
#        self.label = QLabel()
#        self.label.setPixmap(self.im)

#        self.grid = QGridLayout()
#        self.grid.addWidget(self.label, 1, 1)
#        self.setLayout(self.grid)

#        self.setGeometry(50, 50, 320, 200)
#        self.setWindowTitle("PyQT show image")
#        self.show()

    def show_img(self):
        self.ui.frame.set
        label = QLabel()
        pixmap = QPixmap('img.png')
        label.setPixmap(pixmap)

    def new_pic(self):
        ...

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "OpenFile", "", "")
        if image_path:
            print(image_path)
            self.pixmap = QPixmap(image_path)
            self.newImage = ImageView(self.pixmap)
            # self.image_lbl.setPixmap(QPixmap(pixmap))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
