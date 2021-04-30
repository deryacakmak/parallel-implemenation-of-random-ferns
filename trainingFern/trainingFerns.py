import cv2
import random
import numpy as np
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation


PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 12
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


def findCoordinate(A, keypoints):
    
    newKeypoints = []
    
    a00 = A[0]
    a01 = A[1]
    a10 = A[3]
    a11 = A[4]
    t0 = A[2]
    t1 = A[5]   

    for keypoint in keypoints:
        x = keypoint[1]
        y = keypoint[0]
        
        x1 = int(a00*x + a01*y + t0)
        y1 = int(a10*x + a11*y + t1)
        
        newKeypoints.append((x1,y1))
    
    return newKeypoints
    
    

 
def trainingFerns(imageName):
    print("Training started!")
    image = readImage(imageName)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # image = applySmoothing(addNoise(image))
    keypoints = detectKeypoint(image)
    features = initializeClasses(keypoints)

    for i in range(NUM_OF_IMAGES_TO_GENERATES):
       
        warp_dst, matrixM = applyAffineDeformation(image)
        
        newKeypoints = findCoordinate(matrixM.flatten(), keypoints)
        

    #     print("deformed image!",i)
    #     for keypoint in newKeypoints:

    #         classNum = keypoint[0]
    #         y, x = keypoint[1]
    #         index = warp_dst.shape[:2][1]*y+x
    #         patch = findPatch(index, warp_dst.flatten())
    #         features[classNum] = features[classNum] + extractFeature(patch) # [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]
            

    # for i in keypoints:
    #     key = (i[0],i[1])
    #     if len(features[key]) != 0:
    #         ferns = generateFerns(features[key]) # [[1, 1], [0, 1], [0, 0], [1, 0], [0, 0], [0, 1]]

    #         pro = traningClass(ferns) # {2: 3, 3: 2, 1: 1}
    #         features[key] = probablityDistrubition(len(ferns),pro,pow(2,len(ferns[0]))) # (60, 142): {0: 0.2, 1: 0.2, 2: 0.5, -1: 0.11000000000000001}
    # print("Training done!")
    # return features, keypoints


trainingFerns("eiffel_tower.png")





