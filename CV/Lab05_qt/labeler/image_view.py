from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QLabel, QMenu, QMenuBar

from rect import Rect


class ImageView(QWidget):
    def __init__(self, image, img_path):
        super().__init__()
        self.img_path: str = img_path
        self.setWindowTitle("Labeler")
        # layout = QVBoxLayout()
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)

        self.menuBar = QMenuBar(self)
        save = QMenu("&Options")
        save.addAction("Save", self.save_label)
        self.menuBar.addMenu(save)

        self.img = QLabel()
        self.img.setPixmap(image)

        self.rect = Rect()

        self.grid.addWidget(self.menuBar, 0, 0, 1, 2)
        self.grid.addWidget(self.img, 1, 0, 1, 2)
        self.grid.addWidget(self.rect, 1, 0, 1, 2)
        self.show()

    def save_label(self):
        txt_path = self.img_path.rsplit('.', 1)[0]
        with open(f'{txt_path}.txt', mode='w') as f:
            info = self.rect.get_points()
            f.write(f'0 {info}')

    def _createMenuBar(self):
        self.menuBar = QMenuBar(self)
        # self.setMenuBar(menuBar)
        # Creating menus using a QMenu object
        self.fileMenu = QMenu("&File", self)
        self.menuBar.addMenu(self.fileMenu)
        # Creating menus using a title
        editMenu = self.menuBar.addMenu("&Edit")
        helpMenu = self.menuBar.addMenu("&Help")
