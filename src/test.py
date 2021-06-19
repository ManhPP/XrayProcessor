import os

import imageio

from unsharping_mask import UM
from clahe import CLAHE
from hef import HEF


if __name__ == '__main__':
    img = "img.jpg"
    alg = HEF(img)
    processed_image = alg.run()
    filename = "result.jpg"
    imageio.imwrite(filename, processed_image)
