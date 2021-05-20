import math
import cv2
import trainingFerns as tf
import kernelCalls as kc

FERN_NUMBER = 50

def classifyKeypoint(imageName, originalImage):
    
    
    print("Maching started!")
    img = tf.readImage(imageName)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    keypoints = tf.detectKeypoint(img)[:1]
    
    #trainingClasses, originalImageKeypoints, indexList = tf.trainingFerns(originalImage)

    match = []

    for keypoint in keypoints:
        
        # results = [0] * len(originalImageKeypoints)
        
        patch = tf.findPatch(keypoint[0],keypoint[1], img)
        
        print(",,",keypoint[0],keypoint[1])
        
        
        patch2 = kc.matching(img, keypoints, 16);
        
        
        
        cv2.imwrite("patch2Out.png",patch2)
        
        cv2.imwrite("patchOut.png",patch)
        
        
        
        
    #     for i in range(FERN_NUMBER):
            
    #         index = indexList[i]
            
    #         decimalNum = tf.extractFeature(patch.flatten(), index)

    #         for classNum in range(len(originalImageKeypoints)):
                
    #             classCount = trainingClasses[classNum]

    #             if(decimalNum in classCount):
    #                   results[classNum] = math.log(classCount[decimalNum]) + results[classNum] 
                     
    #             # else:
    #             #     results[classNum] = math.log(classCount[-1]) + results[classNum] 
      
    #     max_value = max(results)
    #     max_index = results.index(max_value)
        
    #     match.append((keypoint,originalImageKeypoints[max_index]))
        
        
    # return match
        
classifyKeypoint("eiffel_tower.png","eiffel_tower.png")


