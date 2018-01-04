from PIL import Image, ImageDraw
import numpy as np

from card_finder.model import CardFinder

SAMPLES = 2

if __name__ == '__main__':
    sf = CardFinder()
    sf.init_model()

    tx, ty = sf.dataset_provider.get_batch(SAMPLES)
    py = sf.model.predict(tx)

    for i in range(SAMPLES):
        img = Image.fromarray(np.uint8(tx[i])).convert('RGBA')
        mask = py[i]
        valid_mask = ty[i]

        mask_img = np.dstack((
            mask,
            np.zeros(mask.shape),
            np.zeros(mask.shape),
            np.ones(mask.shape) * 0.5 * mask
        ))
        mask_img = Image.fromarray(np.array(mask_img*255, np.uint8))
        mask_img = mask_img.resize(img.size)

        valid_mask_img = np.dstack((
            np.zeros(valid_mask.shape),
            valid_mask,
            np.zeros(valid_mask.shape),
            np.ones(valid_mask.shape) * 0.25 * valid_mask
        ))
        valid_mask_img = Image.fromarray(np.array(valid_mask_img*255, np.uint8))
        valid_mask_img = valid_mask_img.resize(img.size)

        composed = Image.alpha_composite(img, mask_img)
        composed = Image.alpha_composite(composed, valid_mask_img)

        composed.show()
