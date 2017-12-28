from PIL import Image, ImageDraw
import numpy as np

from card_finder.model import CardFinder

if __name__ == '__main__':
    sf = CardFinder()
    sf.init_model()

    tx, ty = sf.dataset_provider.get_batch(10)
    py = sf.model.predict(tx.reshape(-1, *sf.input_shape))
    pb = py * tuple((*sf.img_size, *sf.img_size) for i in range(len(py)))

    for i, img in enumerate(tx):
        img = Image.fromarray(np.uint8(img))

        x, y, w, h = pb[i]

        d = ImageDraw.Draw(img)
        poly = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        d.polygon(poly, outline=(255, 0, 0))

        x, y, w, h = ty[i]

        poly = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        d.polygon(poly, outline=(0, 255, 0))

        img.show()
