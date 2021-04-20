import math
import trainingFerns as tf


FERN_NUMBER = 11
PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 11
REGULARIZATION_TERM = 1


def calculateProbablity(fern, trainingClasses, classNum):
    decimalNum = tf.convertDecimal(fern)
    trainingClass = trainingClasses[classNum]
    if decimalNum in trainingClass:
        return trainingClasses[classNum][decimalNum]
    else:
        return trainingClasses[classNum][-1]
        

def classifyKeypoint(imageName, originalImage):
    img = tf.readImage(imageName)
    keypoints = tf.detectKeypoint(img)
    trainingClasses = tf.trainingFerns(originalImage)
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
                probability += math.log(calculateProbablity(fern, trainingClasses, i))
            probabilities.append((probability,i))
            
        probabilities.sort(key=lambda x:x[1])
        matchResult.append([keypoint,probabilities[0]])
        break

#classifyKeypoint("3.pgm","3.pgm")

# print(math.ceil(-1.0))


