import cv2
import random
import numpy as np
import math
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation


PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 11
FERN_NUMBER = 11
REGULARIZATION_TERM = 1
NUM_OF_IMAGES_TO_GENERATES = 2

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
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    return np.argwhere(dst > 0.01 * dst.max())

def findCoordinate(x, y, A):
    
     M11 = A[0]
     M12 = A[1]
     M13 = A[2]
     M21 = A[3]
     M22 = A[4]
     M23 = A[5]
     
     xp = M11*x + M12*y + M13;
     yp = M21*x + M22*y + M23;
     
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


def probablityDistrubition(classGraph, K):
    N = sum(classGraph.values())
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
    image = applySmoothing(addNoise(image))
    keypoints = detectKeypoint(image)
    features = initializeClasses(keypoints)
    for i in range(NUM_OF_IMAGES_TO_GENERATES):
        warp_dst, matrixM = applyAffineDeformation(image)
        warp_dst = cv2.cvtColor(warp_dst,cv2.COLOR_BGR2GRAY)
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
        ferns = generateFerns(features[i])
        pro = traningClass(ferns)
        features[i] = probablityDistrubition(pro,pow(2,len(ferns[0])))
   
    return features
        
    
            

trainingFerns("3.pgm")

