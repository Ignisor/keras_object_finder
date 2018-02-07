import abc
import os
import random

import numpy as np
from PIL import Image, ImageDraw


class BaseGenerator(abc.ABC):
    def __init__(self, input_shape=(28, 28, 3), output_shape=(4,), data_amount=10000):
        super().__init__()

        self.amount = data_amount

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.img_w, self.img_h, self.img_ch = self.count_image_sizes()

    def count_image_sizes(self):
        w = h = ch = None
        if len(self.input_shape) == 3:
            w, h, ch = self.input_shape
        if len(self.input_shape) == 2:
            size, ch = self.input_shape
            w = h = size/size

        return w, h, ch

    @abc.abstractmethod
    def _generate(self):
        pass

    def get_batch(self, amount=None):
        amount = amount or self.amount
        imgs = []
        borders = []
        for i in range(amount):
            img, border = self._generate()
            imgs.append(img)
            borders.append(border)

        np_imgs = np.stack(imgs)
        np_borders = np.stack(borders)

        return np_imgs, np_borders

    def get_generator(self):
        while True:
            img, border = self._generate()
            img = img.reshape(-1, *self.input_shape)
            border = border.reshape(-1, *self.output_shape)

            yield img, border

    @staticmethod
    def img_to_numpy(img):
        channels = BaseGenerator.convert_mode_to_channels(img.mode)
        result = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], channels)
        result = result / 255

        return result

    @staticmethod
    def convert_mode_to_channels(mode):
        modes = {
            'L': 1,
            'RGB': 3,
        }

        return modes[mode]


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


class BlackSquaresGenerator(BaseGenerator):
    def _generate(self):
        str_ch = ''
        if self.img_ch == 3:
            str_ch = 'RGB'
        elif self.img_ch == 1:
            str_ch = 'L'

        img = Image.new(str_ch, (self.img_w, self.img_h), color=(255, 255, 255))

        w = random.randint(1, img.width/2)
        h = random.randint(1, img.height/2)
        x = random.randint(0, img.width - w)
        y = random.randint(0, img.height - h)

        draw = ImageDraw.Draw(img)
        draw.rectangle((x, y, x + w, y + h), fill=(0,) * self.img_ch)

        np_img = BlackSquaresGenerator.img_to_numpy(img)

        w = w/img.width
        h = h/img.height
        x = x/img.width
        y = y/img.height

        return np_img, np.array((x, y, w, h))
