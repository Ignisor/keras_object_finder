import numpy as np


class SquareGenerator(object):
    def __init__(self):
        self.img_size = (8, 8)
        self.amount = 10000
        self.min_obj_size = 1
        self.max_obj_size = 5

    def get_batch(self, amount=None):
        return self.generate(amount or self.amount)

    def generate(self, amount):
        imgs = np.zeros((amount, self.img_size[0], self.img_size[1]))
        borders = np.zeros((amount, 4))

        for i_img in range(amount):
            w, h = np.random.randint(self.min_obj_size, self.max_obj_size, size=2)
            x = np.random.randint(0, self.img_size[0] - w)
            y = np.random.randint(0, self.img_size[1] - h)
            imgs[i_img, x:x + w, y:y + h] = 1.0
            borders[i_img] = [x, y, w, h]

        return imgs, borders
