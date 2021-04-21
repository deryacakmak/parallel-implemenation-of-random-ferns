import math
import trainingFerns as tf


FERN_NUMBER = 11
PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 11
REGULARIZATION_TERM = 1


def calculateProbablity(fern, trainingClass, classNum):
    decimalNum = tf.convertDecimal(fern)
    if decimalNum in trainingClass:
        return trainingClass[decimalNum]
    else:
        return trainingClass[-1]


def checkFern(trainingClasses, fern, classNum):
    trainingClass = trainingClasses[classNum]
    if len(trainingClass) != 0:
        return trainingClass
    else:
        return 0

def classifyKeypoint(imageName, originalImage):
    img = tf.readImage(imageName)
    keypoints = tf.detectKeypoint(img)
    trainingClasses, tainingKeypoints= tf.trainingFerns(originalImage)
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
    return matchResult, tainingKeypoints
        

#print(classifyKeypoint("3.pgm","3.pgm"))

# print(math.ceil(-1.0))


