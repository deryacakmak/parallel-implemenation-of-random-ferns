import cv2
import random
import numpy as np
import math
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation

allClasses = dict()
PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 11
FERN_NUMBER = 30
REGULARIZATION_TERM = 1
<<<<<<< Updated upstream
NUM_OF_IMAGES_TO_GENERATES = 2
=======
NUM_OF_IMAGES_TO_GENERATES = 100
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
=======
    image = np.float32(image)
    dst = cv2.cornerHarris(image,2,3,0.04)
>>>>>>> Stashed changes
    return np.argwhere(dst > 0.01 * dst.max())

 
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
        return 1
    else:
        return 0

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
<<<<<<< Updated upstream
    image = applySmoothing(addNoise(image))
=======
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = applySmoothing(addNoise(image))
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        ferns = generateFerns(features[i])
        pro = traningClass(ferns)
        features[i] = probablityDistrubition(pro,pow(2,len(ferns[0])))
   
    return features
        
    
            

trainingFerns("3.pgm")
=======
        if len(features[i]) != 0:
            ferns = generateFerns(features[i])
            #print(ferns)
            
            pro = traningClass(ferns)
            features[i] = probablityDistrubition(pro,pow(2,len(ferns[0])))
    return features, keypoints




image = readImage("eiffel_tower.png")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
index = findPixelIndex(image.shape[:2][1], 5,2)
print(image[5][2],image.flatten()[])
# keypoints = detectKeypoint(image)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# warp_dst, newKeypoints = applyAffineDeformation(image, keypoints.tolist())
# for i in newKeypoints:
#     warp_dst[i[1][0]][i[1][1]] = 255

# cv2.imwrite("findCoordinate.png",warp_dst)





>>>>>>> Stashed changes

