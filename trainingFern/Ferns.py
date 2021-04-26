import math
import cv2
import trainingFerns as tf



PATCH_WIDTH = 32
NUMBER_OF_FEATURE_EVALUATED_PER_PATCH = 12



def calculateProbablity(fern, trainingClass):
    decimalNum = int("".join(str(x) for x in fern), 2)
    if decimalNum in trainingClass:
        return trainingClass[decimalNum]
    else:
        return trainingClass[-1]


def classifyKeypoint(imageName, originalImage):
    print("Maching started!")
    img = tf.readImage(imageName)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    keypoints = tf.detectKeypoint(img)
    trainingClasses, originalImageKeypoints = tf.trainingFerns(originalImage)

    matchResult = []
    
    
    for keypoint in keypoints:
        
        probabilities = []
        index = img.shape[:2][1]*keypoint[0]+keypoint[1]
        patch = tf.findPatch(index, img.flatten(), PATCH_WIDTH)
        features = tf.extractFeature(patch,NUMBER_OF_FEATURE_EVALUATED_PER_PATCH )
        ferns = tf.generateFerns(features)
        
        
        for i in originalImageKeypoints:
            classNum = (i[0], i[1])
            probability = 0
            for fern in ferns:
                trainingClass = trainingClasses[classNum]
                if(len(trainingClass) !=0):
                    probability = probability + math.log(calculateProbablity(fern, trainingClass))
                  
            probabilities.append((probability,classNum))
                      
        probabilities.sort(key=lambda x:x[1])
        matchResult.append([keypoint,probabilities[0]])
    print("Matching done!")
    return matchResult
        

# classifyKeypoint("eiffel_tower.png","eiffel_tower.png")


