import cv2


class Img:
    def __init__(self):
        self._image = None

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, filepath):
        rgb_data = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
        self._image = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2RGBA)

    @property
    def shape(self):
        return self._image.shape
