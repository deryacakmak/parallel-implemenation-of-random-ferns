import cv2
from  FernsParallel import classifyKeypoint
# from trainingFerns import detectKeypoint 

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)



def drawLine(matchResult, originalImagesKeypoints, keypoints, width, img_tmp):
    
    for i in matchResult:
        x1,y1 = keypoints[i][0]
        x2,y2 = originalImagesKeypoints[i][0][0], originalImagesKeypoints[i][0][1] + width
        cv2.line(img_tmp, (y2, x2), (y1, x1), (0, 255, 0), thickness=1)
    cv2.imwrite("resultfinal751.png",img_tmp)
    

matchResult, originalImagesKeypoints, keypoints = classifyKeypoint("eiffel_tower.png","eiffel_tower.png")

img = cv2.imread("concataneta2.png")
img2 = cv2.imread("eiffel_tower.png")



drawLine(matchResult, originalImagesKeypoints, keypoints, img2.shape[:2][1], img)
