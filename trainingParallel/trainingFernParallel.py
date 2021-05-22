import cv2
import random
import numpy as np
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation
import kernelCalls as kc
import calculateTest as c

PATCH_WIDTH = 32
REGULARIZATION_TERM = 1
NUM_OF_IMAGES_TO_GENERATES = 10000
FERN_SIZE = 11
FERN_NUM = 50
K = pow(2,FERN_SIZE)
allIndexList = None
allProbablities = None
allIndexList2 = []


def getPatchSize():
    return PATCH_WIDTH

def getFernSize():
    return FERN_SIZE;


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
    dst = cv2.cornerHarris(image,2,3,0.12)
    return np.argwhere(dst > 0.01 * dst.max())
    

def generateIndex():
    end = (PATCH_WIDTH * PATCH_WIDTH) -1
    indexList = []
    global allIndexList2
    for i in range(FERN_NUM):
        fern = []
        while(len(fern) != FERN_SIZE):
            I1 = random.randint(0, end)
            I2 = random.randint(0, end)
            if abs(I1-I2) >3:
                fern.append([I1, I2])
        indexList  = indexList + fern;
        allIndexList2.append(np.array(fern))
    return  np.array(indexList)

def initializeClasses(keypoints):
    allProbablities = [0] * len(keypoints)
    for i in range(len(keypoints)):
        allProbablities[i] = []
        for j in range(K):
            allProbablities[i].append(0)
    return np.array(allProbablities)


def trainingFerns(imageName):
    print("Training started!")
    global allProbablities
    global allIndexList
    image = readImage(imageName)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # image = applySmoothing(addNoise(image))
    keypoints = detectKeypoint(image)
    allProbablities = initializeClasses(keypoints)
    allIndexList = generateIndex()

    for i in range(NUM_OF_IMAGES_TO_GENERATES):
        
        print("generate image",i)
       
        warp_dst, matrixM = applyAffineDeformation(image)

        newKeypoints = kc.findCoordinate(matrixM, keypoints)
        
        allProbablities = kc.calculateCount(warp_dst, newKeypoints, PATCH_WIDTH, allProbablities, allIndexList, FERN_NUM, FERN_SIZE, REGULARIZATION_TERM)

        # print(allProbablities[2])
        
        # print("*******")
        
        # print((c.trainingFerns(imageName, allIndexList2, keypoints,newKeypoints, warp_dst ))[2])
        
        
        return allProbablities, keypoints, allIndexList

# trainingFerns("eiffel_tower.png")      
        
        
       
        
        
        
        
        
        
        
        
        
        
        

