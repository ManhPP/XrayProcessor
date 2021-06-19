import numpy as np
import cv2 as cv


class CLAHE:
    def __init__(self, filename, n_iter=2, clip_limit=4, window_size=8):
        self.n_iter = n_iter
        self.clip_limit = clip_limit
        self.window_size = window_size
        self.filename = filename

    def run(self):
        img = cv.imread(self.filename, 0)
        # create a CLAHE object (Arguments are optional).
        clahe = cv.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.window_size, self.window_size))
        cl1 = clahe.apply(img)
        # cv.imwrite(self.filename.split(".")[:-1]+"_result." + self.filename.split(".")[-1], cl1)
        return cl1
