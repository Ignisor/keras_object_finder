import numpy as np

from obj_finder.model import SquareFinder


if __name__ == '__main__':
    sf = SquareFinder()
    sf.init_model()
    sf.train(50000, 30)

    tx, ty = sf.dataset_provider.get_batch(10)
    py = sf.model.predict(tx.reshape(-1, 8 * 8))
    pb = py * 8

    boxes = np.zeros((10, 8, 8))
    for n in range(10):
        pred_box = pb[n]
        x = int(round(pred_box[0]))
        y = int(round(pred_box[1]))
        w = int(round(pred_box[2]))
        h = int(round(pred_box[3]))
        boxes[n, x:x + w, y:y + h] = 1.0