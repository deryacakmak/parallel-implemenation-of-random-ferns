import cv2
import numpy as np
import math
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

def readImage(imageName):
    
    image = cv2.imread(imageName)
    
    if image is not None:
        print("\t{} successfully read!".format(imageName))
        return image
    
    print("\t{} is invalid!".format(imageName))
    
    return None


def generateAffineDeformationMatrix():
    
    a, b = np.random.uniform(0.6, 1.5, 2)
    theta, gama = np.random.uniform(0, 30, 2)
    theta = np.radians(theta)
    gama = np.radians(gama)
    
    ctetha, stetha = np.cos(theta), np.sin(theta)
    cgama, sgama = np.cos(gama), np.sin(gama)
    
    Rtetha = np.array(((ctetha, -stetha), (stetha, ctetha)))
    Rgama = np.array(((cgama, -sgama), (sgama, cgama)))
    R2gama = np.array(((cgama, sgama), (-sgama, cgama)))
    
    matrixA = np.dot(np.dot(np.dot(Rtetha, R2gama),[[a,0],[0,b]]),Rgama)

    return matrixA 


def generateAffineDeformationMatrixSIFTForm():
    
    lambdaParameter = 2
    theta = np.random.uniform(-10, 10)
    gama = np.random.uniform(0,180)
    theta = np.radians(theta)
    gama = np.radians(gama)

    ctetha, stetha = np.cos(theta), np.sin(theta)
    cgama, sgama = np.cos(gama), np.sin(gama)
    Rtetha = np.array(((ctetha, -stetha), (stetha, ctetha)))
    Rgama = np.array(((cgama, -sgama), (sgama, cgama)))
    
    t = 1/cgama
    T = [[t,0],[0,1]]
    
    matrixA = lambdaParameter * np.dot(np.dot(Rtetha, T), Rgama)
    
    return matrixA

def calculateMatrixT(A, c, c2):
    
    matrixT = np.dot(A,c)
    
    return np.subtract(c2, matrixT)


def calculateMatrixM(A, T):
    
    a = (2,3)
    M = np.zeros(a)
    M[0][2] = T[0]
    M[1][2] = T[1]
    for i in range(2):
        for j in range(2):
            M[i][j] = A[i][j] 
    return M


def findCorrespondingCornerInTheNewImage(affineMatrix, width, height):
    
    corner1 = np.dot(affineMatrix, [-width/2, -height/2]) # (0,0)
    corner2 = np.dot(affineMatrix, [width/2, -height/2]) # (w,0)
    corner3 = np.dot(affineMatrix, [-width/2, height/2]) # (0,h)
    corner4 = np.dot(affineMatrix, [width/2, height/2]) # (w,h)
    
    corners = [corner1, corner2, corner3, corner4]
    
    return corners 


def findNewImageShape(xCoordinates, yCoordinates):
    
    minx = min(xCoordinates)
    maxx = max(xCoordinates)


    miny = min(yCoordinates)
    maxy = max(yCoordinates)


    newWidth = math.ceil(maxx-minx)
    newHeight = math.ceil(maxy-miny)
    
    return newWidth, newHeight

img = readImage('eiffel_tower.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width = img.shape


affineMatrix = generateAffineDeformationMatrixSIFTForm()
    
corners = findCorrespondingCornerInTheNewImage(affineMatrix, width, height)

xCoordinates = []
yCoordinates = []

for i in corners:
    xCoordinates.append(i[0])
    yCoordinates.append(i[1])

newWidth, newHeight  = findNewImageShape(xCoordinates, yCoordinates)

c1 = [width/2, height/2] # center of original image
c2 = [newWidth/2, newHeight/2] # center of new image

matrixT = calculateMatrixT(affineMatrix, c1, c2)

matrixM = calculateMatrixM(affineMatrix, matrixT).astype(np.float32)
dim_block = 32
dim_grid_x = math.ceil(width / dim_block)
dim_grid_y = math.ceil(height / dim_block)

outImg = np.zeros((newHeight, newWidth )).astype(np.int32)
print(matrixM)

mod = SourceModule("""
__global__ void affineDeformation(float *M, int *in, int *out, int newWidth, int oldWidth, int oldHeight)
  {
    
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
 	int Row = blockIdx.y * blockDim.y + threadIdx.y;
 	
 	
    int newCol = Col * M[0] + Row * M[1] + M[2];
    int newRow = Col * M[3] + Row * M[4] + M[5];
    
    int i = 0;
    
    if(Col < oldWidth && Row < oldHeight){
             i++;
             out[newRow * newWidth + newCol] = in[oldWidth * Row +Col];
            
            }    
    printf("%d",i);

  } """)
  
  
img = img.astype(np.int32)

affineDeformation = mod.get_function("affineDeformation")

print(width*height)
affineDeformation(
        drv.In(matrixM),
        drv.In(img),
        drv.Out(outImg),
        np.int32(newWidth),
        np.int32(width),
        np.int32(height),
        block=(dim_block, dim_block, 1),
        grid=(dim_grid_x, dim_grid_y,1)
    )

