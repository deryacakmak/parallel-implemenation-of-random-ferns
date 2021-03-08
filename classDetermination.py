import numpy as np
import cv2
from skimage.util import random_noise


allKeypoints = dict()

def readImage(imageName):
    
    image = cv2.imread(imageName)
    
    if image is not None:
        print("\t{} successfully read!".format(imageName))
        return image
    
    print("\t{} is invalid!".format(imageName))
    
    return None


def addNoise(image):
    noise_img = random_noise(image, mode='gaussian', seed=None, clip=True)
    return np.array(255*noise_img, dtype = 'uint8')

def applySmoothing(image):
    return cv2.GaussianBlur(image,(7,7),0)

def findCoordinate(x,y,A):
     a00 = A[0]
     a01 = A[1]
     a10 = A[3]
     a11 = A[4]
     t0 = A[2]
     t1 = A[5]
     det_A = a00*a11 - a01*a10
        
     inv_det_A = 1.0 / det_A
     xa = x - t0
     ya = y - t1
     xp = inv_det_A * (a11 * xa - a01 * ya)
     yp = inv_det_A * (a00 * ya - a10 * xa)
     return int(xp), int(yp)
 
def detectKeypoint(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    return np.argwhere(dst > 0.01 * dst.max())

def detectKeypointOriginalImage(image):
    blur_image = applySmoothing(addNoise(image))
    keypoints = detectKeypoint(blur_image)
    for i in keypoints:
        i = tuple(i)
        allKeypoints[(i[0],i[1])] = 0
        

def updateKeypointDict(keypoints,A):
    for i in keypoints:
        print(i)
        x , y = findCoordinate(i[0],i[1],A) 
        point = (x,y)
        #print(point)
        if point in allKeypoints:
            allKeypoints[(x,y)] +=1
        else:
            allKeypoints[(x,y)] = 0


img = readImage("eiffel_tower.png")
detectKeypointOriginalImage(img)


A = [ 0.89122576,   1.0619898,    0.34270653, 0.83097535,  -0.57304543, 241.54305]
A = np.array(A).astype(np.float32)

img2 = cv2.imread('0.png')
keypoints2 = detectKeypoint(img2)
updateKeypointDict(keypoints2,A)


