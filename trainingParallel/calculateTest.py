import cv2
import random
import numpy as np
from skimage.util import random_noise
from affineDeformation import applyAffineDeformation



PATCH_WIDTH = 32
REGULARIZATION_TERM = 1
NUM_OF_IMAGES_TO_GENERATES = 1
FERN_SIZE = 3
FERN_NUM = 5
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
    dst = cv2.cornerHarris(image,2,3,0.12)
    return np.argwhere(dst > 0.01 * dst.max())



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

def checkIntensityOfPixel(I1, I2):
    if I1 <  I2:
        return 1
    else:
        return 0
    

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

# TODO : index shift should be added!
def extractFeature(patch, index):
    lenght = len(patch)
    features = []
 
    for i in index:
        
        if i[0] < lenght and i[1] < lenght:
            features.append(checkIntensityOfPixel(patch[i[0]], patch[i[1]]))
    if len(features) != 0:

        
        return int("".join(str(x) for x in features), 2)
    else:
        return -1


def probablityDistrubition(classGraph, K):
    values = classGraph.values()
    N = sum(values)
    classGraph[-1] = 0
    for k in classGraph:
        Nkc = classGraph[k]
        classGraph[k] = (Nkc + REGULARIZATION_TERM) / (N + K * REGULARIZATION_TERM)
    return classGraph

# TODO: for K should be deleted!
def initializeClasses(keypoints):
    allProbablities = [0] * len(keypoints)
    for i in range(len(keypoints)):
        allProbablities[i] = dict()
        for j in range(K):
            allProbablities[i][j] = 0
        allProbablities[i][-1] = 0
    return np.array(allProbablities)


def findCoordinate(A, keypoints):
    
    
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
        
        x1 = int(a00*x + a01*y + t0)
        y1 = int(a10*x + a11*y + t1)
        
        newKeypoints.append((x1,y1))
    
    return newKeypoints
    

    
    
def calculateCount(patch, classNum):
    classCount = allProbablities[classNum]
    for i in range(FERN_NUM):
        index = allIndexList[i]
        decimalNum = extractFeature(patch, index)
        #print(decimalNum)
        if decimalNum != -1:
            if(decimalNum in classCount):
                classCount[decimalNum] =  classCount[decimalNum] + 1
            else:
                classCount[decimalNum] = 1

    
 
def trainingFerns(imageName, allIndexList2, keypoints,newKeypoints, warp_dst ):
    print("Training started!")
    global allProbablities
    global allIndexList
    image = readImage(imageName)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # image = applySmoothing(addNoise(image))
    keypoints = keypoints
    allProbablities = initializeClasses(keypoints)
    allIndexList = allIndexList2

    for i in range(NUM_OF_IMAGES_TO_GENERATES):
        
        print("generate image",i)
        
        #warp_dst, matrixM = applyAffineDeformation(image)
        
        #newKeypoints = findCoordinate(matrixM.flatten(), keypoints)
      
        # del matrixM

        # print("deformed image!",i)
        
        classNum = 0
        for keypoint in newKeypoints:
            
            patch = findPatch(keypoint[1],keypoint[0], warp_dst).flatten()
            
            calculateCount(patch, classNum)
        
            classNum +=1
        del warp_dst
            
    # for i in range(len(allProbablities)):
        
    #     allProbablities[i] = probablityDistrubition(allProbablities[i],K)

    
    print("Training done!")
    return allProbablities


# trainingFerns("eiffel_tower.png")



        
        # for keypoint in newKeypoints:
        #     warp_dst[keypoint[1]][keypoint[0]] = 255
            
        # cv2.imwrite("affineTest45.png",warp_dst)




# image = readImage("eiffel_tower.png")
# image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# patch = image[-100:32,80:112]

# cv2.imwrite("resuldfdt.png",patch)

# keypoints = detectKeypoint(image)

# x, y = keypoints[5]

# image[y][x] = 255

# path = findPatch(y, x, image)



# cv2.imwrite("result.png",patch)







