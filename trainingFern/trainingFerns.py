import cv2
import random
import numpy as np
import math
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation

# key: class num 
# value: (0...n, P(F|C)) 
allClasses = dict()

PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 11
FERN_NUMBER = 11
REGULARIZATION_TERM = 1
NUM_OF_IMAGES_TO_GENERATES = 1

def readImage(imageName):
    image = cv2.imread(imageName)
    if image is not None:
        print("\t{} successfully read!".format(imageName))
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("\t{} is invalid!".format(imageName))
    return None


def addNoise(image):
    noise_img = random_noise(image, mode='gaussian', seed=None, clip=True)
    return np.array(255*noise_img, dtype = 'uint8')

def applySmoothing(image):
    return cv2.GaussianBlur(image,(7,7),0)

def detectKeypoint(image):
    image = np.float32(image)
    dst = cv2.cornerHarris(image,2,3,0.04)
    return np.argwhere(dst > 0.01 * dst.max())

def findCoordinate(x, y, A):
    
     a00 = A[0]
     a01 = A[1]
     a10 = A[3]
     a11 = A[4]
     t0 = A[2]
     t1 = A[5]
     
     xp = a00 * x + a01 * y + t0
     yp = a10 * x + a11 * y + t1
     
     return int(xp), int(yp)
 
def findPixelIndex(width, x, y):
    return width*y+x
    
    
def findPatch(pixelIndex, image):
    start = int(pixelIndex- (PATCH_WIDTH/2))
    end = int(pixelIndex + (PATCH_WIDTH/2))
    if start < 0: 
        start = 0 
    if end > image.shape[0]: 
        end = image.shape[0]
    return image[start:end]

def checkIntensityOfPixel(I1, I2):
    if I1 > I2:
        return 0
    else:
        return 1

def extractFeature(patch):
    features = []
    end = patch.shape[0] -1
    while(len(features) != NUMBER_OF_FEATURE_EVALUATED_PER_PATCH):
        I1 = random.randint(0, end)
        I2 = random.randint(0, end)
        if abs(I1-I2) >3:
            features.append(checkIntensityOfPixel(I1, I2))
    return features


# tüm resimlerden gelen keypointe ait featureları fernlere böl
def generateFerns(features):
    S = math.ceil(len(features)/FERN_NUMBER)
    random.shuffle(features)
    return [features[x:x+S] for x in range(0, len(features), S)]

def convertDecimal(fern):
    return int("".join(str(x) for x in fern), 2)

# [[0,1,0], [0,1,0], [0,1,0]]
def traningClass(ferns):
    classGraph = dict()
    for fern in ferns:
        num = convertDecimal(fern)
        if num in classGraph:
            classGraph[num] = classGraph[num] + 1
        else:
            classGraph[num] = 1
    return classGraph

def trainingFerns(imageName):
    
    image = readImage(imageName)
    image = applySmoothing(addNoise(image))
    keypoints = detectKeypoint(image)
    classNum = 1
    for keypoint in keypoints:
        features = []
        for i in range(NUM_OF_IMAGES_TO_GENERATES):
            deformedImage, matrixM = applyAffineDeformation(image)
            x, y = findCoordinate(keypoint[0], keypoint[1], matrixM.flatten())
            index = findPixelIndex(deformedImage.shape[:2][1], x, y)
            patch = findPatch(index, deformedImage.flatten())
            features = features + extractFeature(patch)
        ferns = generateFerns(features)
        probablityDistrubition = traningClass(ferns)
        allClasses[classNum] = probablityDistrubition
        classNum +=1
            
        
