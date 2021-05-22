import math
import cv2
import trainingFernParallel as tf
import kernelCalls as kc

FERN_NUMBER = 50


def classifyKeypoint(imageName, originalImage):
    
    
    print("Maching started!")
    img = tf.readImage(imageName)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    keypoints = tf.detectKeypoint(img)

    allProbablities, originalImageKeypoints, indexList = tf.trainingFerns(originalImage)

    matchResult = kc.matching(img, keypoints, tf.getPatchSize() , allProbablities,indexList, FERN_NUMBER, tf.getFernSize())
               
    return matchResult, originalImageKeypoints, keypoints
       
# classifyKeypoint("eiffel_tower.png","eiffel_tower.png")


