import cv2
import numpy as np


def readImage(imageName):
    image = cv2.imread(imageName)
    if image is not None:
        print("\t{} successfully read!".format(imageName))
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("\t{} is invalid!".format(imageName))
    return None



def detectKeypoint(image):
    image = np.float32(image)
    dst = cv2.cornerHarris(image,2,3,0.04)
    return np.argwhere(dst > 0.01 * dst.max())


