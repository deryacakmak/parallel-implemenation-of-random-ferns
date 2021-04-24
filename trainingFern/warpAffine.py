
def warpAffine(A, inputImage,  newWidth,  newHeight , oldWidth,  oldHeight, outputImage, keypointList):
    
    keypoints = []
    
 
    for Row in range(newHeight):
       
        for Col in range(newWidth):
             
             a00 = A[0]
             a01 = A[1]
             a10 = A[3]
             a11 = A[4]
             t0 = A[2]
             t1 = A[5]
             det_A = a00*a11 - a01*a10
             
             inv_det_A = 1.0 / det_A
             
             xa = Col - t0;
             ya = Row - t1;
             xp = inv_det_A * (a11 * xa - a01 * ya);
             yp = inv_det_A * (a00 * ya - a10 * xa);
               
             value = 0;
             
             xi = int(xp)
             yi = int(yp)
             
             if check([yi,xi], keypointList):
                 keypointList.remove([yi,xi])
                 keypoints.append([(yi,xi),(Row,Col)])
             
             if (xi >= 0 and yi >= 0 and xi < oldWidth-1 and yi < oldHeight-1):
                 
                 I00 = inputImage[yi][xi]
                 I10 = inputImage[yi+1][xi]
                 I01 = inputImage[yi][xi+1]
                 I11 = inputImage[yi+1][xi+1]
                 alpha = xp - xi
                 beta = yp - yi
                 
                 interp = (1.0 - alpha) * (1.0 - beta) * I00 + (1.0 - alpha) * beta * I01  + alpha * (1.0 - beta) * I10 + alpha * beta * I11

                 interpi = int(interp)
                
                 if (interpi < 0):
                        interpi = 0
                 elif (interpi > 255):
                        interpi = 255
                 
                 value = interpi
            
             outputImage[Row][Col] = value
             
    return outputImage, keypoints


def check(coordinants, keypointList):
    for keypoint in keypointList:
        if(coordinants[0] == keypoint[0] and coordinants[1] == keypoint[1]  ):
            return True
    return False

