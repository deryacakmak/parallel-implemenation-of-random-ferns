import cv2
import random
import numpy as np
import math
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation


PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 11
FERN_NUMBER = 30
REGULARIZATION_TERM = 1
NUM_OF_IMAGES_TO_GENERATES = 100

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

 
def findPixelIndex(width, x, y):
    return width*y+x
    
    
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
def generateFerns(features, fernNumber = FERN_NUMBER):
    S = math.ceil(len(features)/fernNumber)
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


def probablityDistrubition(classGraph, K):
    N = sum(classGraph.values())
    classGraph[-1] = (REGULARIZATION_TERM) / (N + K * REGULARIZATION_TERM) 
    for k in classGraph:
        Nkc = classGraph[k]
        classGraph[k] = (Nkc + REGULARIZATION_TERM) / (N + K * REGULARIZATION_TERM)
    return classGraph


def initializeClasses(keypoints):
    features = dict()
    for i in range(len(keypoints)):
        features[i] = []
    return features
        

 
def trainingFerns(imageName):
    
    image = readImage(imageName)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = applySmoothing(addNoise(image))
    keypoints = detectKeypoint(image)
    features = initializeClasses(keypoints)

    for i in range(NUM_OF_IMAGES_TO_GENERATES):
        warp_dst, newKeypoints = applyAffineDeformation(image, keypoints.tolist())
        height, width = warp_dst.shape[:2]
        classNum = 0
        for keypoint in keypoints:
            x, y = findCoordinate(keypoint[0], keypoint[1], matrixM.flatten())
            if(x<width and y<height):
                index = findPixelIndex(width, x, y)
                patch = findPatch(index, warp_dst.flatten())
                features[classNum] = features[classNum] + extractFeature(patch)
            classNum +=1

    for i in range(len(keypoints)):
        if len(features[i]) != 0:
            ferns = generateFerns(features[i])
            #print(ferns)
            
            pro = traningClass(ferns)
            features[i] = probablityDistrubition(pro,pow(2,len(ferns[0])))
    return features, keypoints




image = readImage("eiffel_tower.png")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
index = findPixelIndex(image.shape[:2][1], 5,2)
print(image[5][2],image.flatten()[index])
# keypoints = detectKeypoint(image)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# warp_dst, newKeypoints = applyAffineDeformation(image, keypoints.tolist())
# for i in newKeypoints:
#     warp_dst[i[1][0]][i[1][1]] = 255

# cv2.imwrite("findCoordinate.png",warp_dst)






