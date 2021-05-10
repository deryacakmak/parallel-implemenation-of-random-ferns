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
    