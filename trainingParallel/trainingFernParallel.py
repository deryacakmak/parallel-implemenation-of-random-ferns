import cv2
import random
import numpy as np
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation
import kernelCalls as kc

PATCH_WIDTH = 32
REGULARIZATION_TERM = 1
NUM_OF_IMAGES_TO_GENERATES = 1
FERN_SIZE = 11
FERN_NUM = 50
K = pow(2,FERN_SIZE)
allIndexList = None
allProbablities = None

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


def generateIndex():
    end = (PATCH_WIDTH * PATCH_WIDTH) -1
    indexList = []
    for i in range(FERN_NUM):
        fern = []
        while(len(fern) != FERN_SIZE):
            I1 = random.randint(0, end)
            I2 = random.randint(0, end)
            if abs(I1-I2) >3:
                fern.append([I1, I2])
        indexList.append(np.array(fern))
    return  np.array(indexList)

def initializeClasses(keypoints):
    allProbablities = [0] * len(keypoints)
    for i in range(len(keypoints)):
        allProbablities[i] = dict()
        for j in range(K):
            allProbablities[i][j] = 0
        allProbablities[i][-1] = 0
    return np.array(allProbablities)


def findCoordinate2(A, keypoints):
    
    
    newKeypoints = []
    a00 = A[0]
    a01 = A[1]
    a10 = A[3]
    a11 = A[4]
    t0 = A[2]
    t1 = A[5]   

    for i in  range(len(keypoints)):
        x = keypoints[i][1]
        y = keypoints[i][0]
        
        print(x,y)
        
        x1 = int(a00*x + a01*y + t0)
        y1 = int(a10*x + a11*y + t1)
        
        newKeypoints.append((x1,y1))
    
    return newKeypoints

def findPatch(x, y, image, patchWidth = PATCH_WIDTH):

    patchSize =  int(PATCH_WIDTH /2)
    width, height = image.shape[:2]
    
    startX = x - patchSize
    endX =  x + patchSize

    startY = y - patchSize
    endY = y + patchSize

    if(startX < 0  ):
        startX = 0
        
    if (endX >= width ):
        endX = width -1
        
    if(startY < 0 ):
        startY = 0
        
    if (endY >= height):
        endY = height -1

    return image[startX:endX, startY:endY]


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

        newKeypoints = kc.findCoordinate(matrixM, keypoints)[:1]
        
        print()
        
        out = kc.calculateCount(warp_dst, newKeypoints, PATCH_WIDTH)
        
        
        
        image2 = findPatch(newKeypoints[0][1], newKeypoints[0][0], warp_dst)
 
        print(image2)
        
        cv2.imwrite("resultfinal7514.png",image2)
        cv2.imwrite("resultfinal7514sdvd.png",out)
        
        
        
        # for i in newKeypoints:
        #   warp_dst[i[1]][i[0]] = 255
        # cv2.imwrite("resultfinal751.png",warp_dst)
        
        
        # for i in keypoints:
        #   image[i[0]][i[1]] = 255
        # cv2.imwrite("resultfinal7514.png",image)

        
        
        
trainingFerns("eiffel_tower.png")      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

