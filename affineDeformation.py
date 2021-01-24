import cv2
import numpy as np
import math
import time

# [x', y'] = A[x, y] + T

# T = [t1 t2]

# M = [A T]



NUM_OF_IMAGES_TO_GENERATES = 1

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
    

    return np.dot(np.dot(np.dot(Rtetha, R2gama),[[a,0],[0,b]]),Rgama)


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
    
    
    return lambdaParameter * np.dot(np.dot(Rtetha, T), Rgama)

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



def getNewImageShape(affineMatrix, width, height):
    corners = findCorrespondingCornerInTheNewImage(affineMatrix, width, height)

    xCoordinates = []
    yCoordinates = []

    for i in corners:
        xCoordinates.append(i[0])
        yCoordinates.append(i[1])

    return findNewImageShape(xCoordinates, yCoordinates)


def generateAffineMatrix():
    affineMatrixes = []
    for i in range(NUM_OF_IMAGES_TO_GENERATES):
        affineMatrixes.append(generateAffineDeformationMatrixSIFTForm())
    return affineMatrixes
    
def main():
    

    print("->Loading images")
    
    img = readImage('eiffel_tower.png')
    
    height, width = img.shape[:2]
    
    
    start = time.time()
    
    numOfImage = 0
    
    affineMatrixes = generateAffineMatrix()
    
    while numOfImage != NUM_OF_IMAGES_TO_GENERATES:
        
        affineMatrix =affineMatrixes[numOfImage]
    
        newWidth, newHeight = getNewImageShape(affineMatrix, width, height)

        c1 = [width/2, height/2] # center of original image
        c2 = [newWidth/2, newHeight/2] # center of new image

        matrixT = calculateMatrixT(affineMatrix, c1, c2)

        matrixM = calculateMatrixM(affineMatrix, matrixT).astype(np.float32)
        
        try: 
            warp_dst = cv2.warpAffine(img, matrixM, (newWidth, newHeight))
            # if(numOfImage<1250):
            # 	cv2.imwrite("/home/derya/Desktop/affineDeformation/outputs/"+str(numOfImage)+".png", warp_dst)
            # else:
            #     cv2.imwrite("/home/derya/Desktop/affineDeformation/outputs2/"+str(numOfImage)+".png", warp_dst)
            numOfImage = numOfImage +1
            del warp_dst
            del affineMatrix
            del matrixT
            del matrixM
        except:
            del affineMatrix
            del matrixT
            del matrixM
            continue
        
    
        # cv2.imshow('Output', warp_dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #gc.collect()
    

    
    
    end = time.time()
    print(f"Runtime of the program is {end - start}")

if __name__ == "__main__":
    main()





