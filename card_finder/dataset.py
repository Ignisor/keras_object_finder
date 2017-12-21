import os
import random

import numpy as np
from PIL import Image


class CardImgGenerator(object):
    def __init__(self):
        self.img_size = (8, 8)
        self.amount = 10000
        self.min_obj_size = 1
        self.max_obj_size = 5

        self.backgrounds_dir = 'card_finder/src_images/backgrounds/'
        self.cards_dir = 'card_finder/src_images/cards/'

    def get_batch(self, amount=None):
        return self.generate(amount or self.amount)

    def generate(self, amount):
        bg = self.get_background_img()
        card = self.get_card_img()

        result = Image.alpha_composite(bg, card)

        return result

    def get_background_img(self):
        fname = random.choice(os.listdir(self.backgrounds_dir))
        fpath = self.backgrounds_dir + fname

        img = Image.open(fpath)
        img = img.convert('RGBA')

        return img

    def get_card_img(self):
        fname = random.choice(os.listdir(self.cards_dir))
        fpath = self.cards_dir + fname

        img = Image.open(fpath)

        return img
