import cv2
from scipy import ndimage
import Ferns as ferns

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)



def drawLine(keypointListOriginal, keypointListMatch, width, img_tmp):
    for i in range(len(keypointListMatch)):
        x1, y1 = keypointListMatch[i][0][0]+width, keypointListMatch[i][0][1]
        classNum = keypointListMatch[1][1][1]
        x2, y2 = keypointListOriginal[classNum][0], keypointListOriginal[classNum][1]
        cv2.line(img_tmp, (x2, y2), (x1, y1), (0, 255, 0), thickness=2)

    cv2.imwrite("result.png",img_tmp)
    
img = cv2.imread("eiffel_tower.png")
height, width = img.shape[:2]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("eiffel.png")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

v_img = hconcat_resize_min([gray,img2])

matchResult, trainingKeypoints = ferns.classifyKeypoint("eiffel.png", "eiffel_tower.png")
#print(matchResult[0][0])
# keypointListMatch = []
# for i in matchResult:
#     keypointListMatch.append(i[0])
drawLine(trainingKeypoints, matchResult, width, v_img)



# img = cv2.imread("eiffel.png")
# cv2.line(img, (width, height),(width,5),  (0, 255, 0), thickness=1)
# cv2.imwrite("result.png",img)



# rotated = ndimage.rotate(gray, 4)

# cv2.imwrite("eiffel.png",rotated)