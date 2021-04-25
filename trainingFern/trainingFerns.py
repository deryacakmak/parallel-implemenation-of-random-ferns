import cv2
import random
import numpy as np
import math
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation


PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 12
FERN_NUMBER = 2
REGULARIZATION_TERM = 1
NUM_OF_IMAGES_TO_GENERATES = 1
FERN_SIZE = 2

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

def detectKeypoint(image):
    image = np.float32(image)
    dst = cv2.cornerHarris(image,2,3,0.04)
    return np.argwhere(dst > 0.01 * dst.max())


def findPatch(pixelIndex, image, patchWidth = PATCH_WIDTH):
    start = int(pixelIndex- (patchWidth/2))
    end = int(pixelIndex + (patchWidth/2))
    if start < 0: 
        start = 0 
    if end > image.shape[0]: 
        end = image.shape[0]
    return image[start:end]

def checkIntensityOfPixel(I1, I2):
    if I1 > I2:
        return 1
    else:
        return 0

def extractFeature(patch, numberOfFeatureEvaluatedPerPatch = NUMBER_OF_FEATURE_EVALUATED_PER_PATCH):
    features = []
    end = patch.shape[0] -1
    while(len(features) != numberOfFeatureEvaluatedPerPatch):
        I1 = random.randint(0, end)
        I2 = random.randint(0, end)
        if abs(I1-I2) >3:
            features.append(checkIntensityOfPixel(patch[I1], patch[I2]))
    return features


# tüm resimlerden gelen keypointe ait featureları fernlere böl
def generateFerns(features):
    random.shuffle(features)        
    return [features[x:x+FERN_SIZE] for x in range(0, len(features), FERN_SIZE)]


# [[0,1,0], [0,1,0], [0,1,0]]
def traningClass(ferns):
    classGraph = dict()
    for fern in ferns:
        num = int("".join(str(x) for x in fern), 2)
        if num in classGraph:
            classGraph[num] = classGraph[num] + 1
        else:
            classGraph[num] = 1
    return classGraph


def probablityDistrubition(N,classGraph, K):
    classGraph[-1] = (REGULARIZATION_TERM) / (N + K * REGULARIZATION_TERM) 
    for k in classGraph:
        Nkc = classGraph[k]
        classGraph[k] = (Nkc + REGULARIZATION_TERM) / (N + K * REGULARIZATION_TERM)
    return classGraph


def initializeClasses(keypoints):
    features = dict()
    for i in keypoints:
        key = (i[0],i[1])
        features[key] = []
    return features
        

 
def trainingFerns(imageName):
    
    image = readImage(imageName)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # image = applySmoothing(addNoise(image))
    keypoints = detectKeypoint(image)
    features = initializeClasses(keypoints)

    for i in range(NUM_OF_IMAGES_TO_GENERATES):
        warp_dst, newKeypoints = applyAffineDeformation(image, keypoints.tolist())
        for i in newKeypoints:
            image[i[0][0]][i[0][1]] = 255


        for keypoint in newKeypoints:
            classNum = keypoint[0]
            y, x = keypoint[1]
            index = warp_dst.shape[:2][1]*y+x
            patch = findPatch(index, warp_dst.flatten())
            features[classNum] = features[classNum] + extractFeature(patch)


    for i in keypoints:
        key = (i[0],i[1])
        if len(features[key]) != 0:
            ferns = generateFerns(features[key])
            pro = traningClass(ferns)
            features[key] = probablityDistrubition(len(ferns),pro,pow(2,len(ferns[0])))
    return features, keypoints


# trainingFerns("eiffel_tower.png")




