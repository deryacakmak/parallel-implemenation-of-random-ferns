import cv2
from scipy import ndimage
from  Ferns import classifyKeypoint

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)



def drawLine(keypointListOriginal, keypointListMatch, width, img_tmp):
    for i in range(len(keypointListMatch)):
        x1, y1 = keypointListMatch[i][0][0], keypointListMatch[i][0][1]+width
        classNum = keypointListMatch[1][1][1]
        x2, y2 = keypointListOriginal[classNum][0], keypointListOriginal[classNum][1]
        cv2.line(img_tmp, (x2, y2), (x1, y1), (0, 255, 0), thickness=2)

    cv2.imwrite("result.png",img_tmp)
    

matchResult = classifyKeypoint("rotatedImage.png","eiffel_tower.png")

