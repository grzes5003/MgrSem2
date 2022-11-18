from PySide6.QtWidgets import QWidget, QLabel, QMenu, QMenuBar

from ui_image_viewer import Ui_Form

class ImageView(QWidget):
    def __init__(self, image):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.label.setPixmap(image)
        self.ui.label.resize(image.width(), image.height())
        self.resize(image.width() + 10, image.height() + 10)

        self.menuBar = QMenuBar(self)
        fileMenu = QMenu("&File", self)

        self._createMenuBar()

        self.show()

    def _createMenuBar(self):
        self.menuBar = QMenuBar(self)
        # self.setMenuBar(menuBar)
        # Creating menus using a QMenu object
        self.fileMenu = QMenu("&File", self)
        self.menuBar.addMenu(self.fileMenu)
        # Creating menus using a title
        editMenu = self.menuBar.addMenu("&Edit")
        helpMenu = self.menuBar.addMenu("&Help")
