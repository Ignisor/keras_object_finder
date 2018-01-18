import os
import random

import numpy as np
from PIL import Image


class CardImgGenerator(object):
    def __init__(self):
        self.img_size = (128, 128)
        self.back_size = (600, 450)
        self.card_width_range = (100, 300)
        self.mask_shape = (10, 10)

        self.amount = 10000

        self.input_shape = (self.img_size[0], self.img_size[1], 3)
        self.output_shape = (100, )

        self.backgrounds_dir = 'card_finder/src_images/backgrounds/'
        self.cards_dir = 'card_finder/src_images/cards/'

    def get_batch(self, amount=None):
        imgs = np.zeros((amount, self.input_shape[0], self.input_shape[1], 3))
        masks = np.zeros((amount, *self.output_shape))

        for i in range(amount):
            img, mask = self.generate_img()
            imgs[i] = img.reshape(-1, *self.input_shape)

            masks[i] = mask.reshape(self.output_shape)

        return imgs, masks

    def get_generator(self):
        while True:
            img, mask = self.generate_img()
            img = img.reshape(-1, *self.input_shape)
            mask = mask.reshape(-1, *self.output_shape)

            yield img, mask

    def generate_img(self):
        bg = self.get_background_img()
        card = self.get_card_img()

        card_w = random.randint(*self.card_width_range)
        card_h = int((card_w / card.width) * card.height)

        card = card.resize((card_w, card_h))

        offset = (
            random.randint(0, bg.width - card.width),
            random.randint(0, bg.height - card.height)
        )

        img = bg
        img.paste(card, offset, card)
        img = img.resize(self.img_size)
        result = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
        result = result/255

        mask = np.zeros(self.mask_shape)
        card_center = (offset[0] + (card_w // 2), offset[1] + (card_h // 2))
        card_center = (card_center[0]/self.back_size[0], card_center[1]/self.back_size[1])
        mask[round(card_center[1] * 10)][round(card_center[0] * 10)] = 1

        return result, mask

    def get_background_img(self):
        fname = random.choice(os.listdir(self.backgrounds_dir))
        fpath = self.backgrounds_dir + fname

        img = Image.open(fpath)
        img = img.resize(self.back_size)

        return img

    def get_card_img(self):
        fname = random.choice(os.listdir(self.cards_dir))
        fpath = self.cards_dir + fname

        img = Image.open(fpath)

        return img
