import imageio
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage import img_as_float


class UM:
    def __init__(self, filename, filter_type=1, radius=5, amount=2):
        self.amount = amount
        self.filter = filter_type
        self.radius = radius
        self.filename = filename

    def run(self):
        image = imageio.imread(self.filename)
        image = img_as_float(image)

        if self.filter == 1:
            blurred_image = gaussian_filter(image, sigma=self.radius)
        else:
            blurred_image = median_filter(image, size=20)

        mask = image - blurred_image
        sharpened_image = image + mask * self.amount

        sharpened_image = np.clip(sharpened_image, float(0), float(1))
        sharpened_image = (sharpened_image * 255).astype(np.uint8)

        return sharpened_image
