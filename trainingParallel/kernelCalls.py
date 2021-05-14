import pycuda.autoinit
import pycuda.compiler as compiler
import pycuda.driver as drv
import numpy as np



def findCoordinate(matrixM, keypoints):
    
    dim_block = len(keypoints)
    
    
    newKeypoints = np.zeros((len(keypoints), 2 )).astype(np.int32)
    
    keypoints = keypoints.astype(np.int32)
    
    matrixK = matrixM.astype(np.float32)
    
    mod = compiler.SourceModule(open('findCoordinate.cu').read())
            
    affineDeformation = mod.get_function("findCoordinate")
    affineDeformation(
    drv.In(matrixK),
    drv.In(keypoints),
    drv.Out(newKeypoints),
    np.int32(len(keypoints)),
    block=(dim_block, 1, 1),
    grid=(1, 1,1),
        )
    
    return newKeypoints



def calculateCount(image, keypoints, patchSize, allProbablities, allIndexList):
    
    dim_block = len(keypoints)
    
    print(allIndexList)
    
    out = np.zeros((patchSize,patchSize)).astype(np.uint8)
    
    
    patchSize = int(patchSize/2)
    
      
    mod = compiler.SourceModule(open('calculateCount.cu').read())
    
    width, height = image.shape[:2]
    
    image = image.astype(np.uint8)
    
    allIndexList = allIndexList.astype(np.int32)
            
    affineDeformation = mod.get_function("calculateCount")
    affineDeformation(
    drv.In(keypoints),
    drv.In(image),
    drv.In(allProbablities),
    drv.In(allIndexList),
    np.int32(patchSize),
    np.int32(width),
    np.int32(height),
    block=(dim_block, 1, 1),
    grid=(1, 1,1),
        )
    
    return out