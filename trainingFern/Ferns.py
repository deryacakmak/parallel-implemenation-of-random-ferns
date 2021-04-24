import cv2
import numpy as np


def readImage(imageName):
    image = cv2.imread(imageName)
    if image is not None:
        print("\t{} successfully read!".format(imageName))
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("\t{} is invalid!".format(imageName))
    return None



<<<<<<< Updated upstream
def detectKeypoint(image):
    image = np.float32(image)
    dst = cv2.cornerHarris(image,2,3,0.04)
    return np.argwhere(dst > 0.01 * dst.max())
=======

def checkFern(trainingClasses, fern, classNum):
    trainingClass = trainingClasses[classNum]
    if len(trainingClass) != 0:
        return trainingClass
    else:
        return 0

def classifyKeypoint(imageName, originalImage):
    img = tf.readImage(imageName)
    keypoints = tf.detectKeypoint(img)
    trainingClasses, trainingKeypoints= tf.trainingFerns(originalImage)
    totalClassNum = len(trainingClasses)
    matchResult = []
    for keypoint in keypoints:
        probabilities = []
        index = tf.findPixelIndex(img.shape[:2][1], keypoint[0], keypoint[1])
        patch = tf.findPatch(index, img.flatten(), PATCH_WIDTH)
        features = tf.extractFeature(patch,NUMBER_OF_FEATURE_EVALUATED_PER_PATCH )
        ferns = tf.generateFerns(features, FERN_NUMBER)
        for i in range(0,totalClassNum):
            probability = 0
            for fern in ferns:
                trainingClass = checkFern(trainingClasses, fern, i) 
                if(trainingClass != 0):
                    probability += math.log(calculateProbablity(fern, trainingClass, i))
            probabilities.append((probability,i))
        probabilities.sort(key=lambda x:x[1])
        matchResult.append([keypoint,probabilities[0]])
    return matchResult, trainingKeypoints
        

#print(classifyKeypoint("3.pgm","3.pgm"))

# print(math.ceil(-1.0))
>>>>>>> Stashed changes


