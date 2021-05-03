import cv2
from  Ferns import classifyKeypoint
# from trainingFerns import detectKeypoint 

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)



def drawLine(matchResult, width, img_tmp):
    
    for i in matchResult:
        x1,y1 = i[1]
        x2,y2 = i[0][0], i[0][1] + width
        cv2.line(img_tmp, (y2, x2), (y1, x1), (0, 255, 0), thickness=1)
    cv2.imwrite("result.png",img_tmp)
    

matchResult = classifyKeypoint("rotatedImage.png","eiffel_tower.png")

img = cv2.imread("concatanate.png")
img2 = cv2.imread("eiffel_tower.png")

# img3 = cv2.imread("rotatedImage.png")

# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# k1 = detectKeypoint(img3)
# k2 = detectKeypoint(img2)
# matchResult = []
# for i in range(len(k1)):
#     matchResult.append((k1[i],k2[i]))





# for i in matchResult:
#     x,y = i[1][1]
#     img[x][y] = 255

# cv2.imwrite("result.png",img)

drawLine(matchResult, img2.shape[:2][1], img)