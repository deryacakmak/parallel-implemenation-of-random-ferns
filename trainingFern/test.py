import cv2
from scipy import ndimage


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)



def drawLine(keypointListOriginal, keypointListMatch, width, img_tmp):
    for i in range(len(keypointListMatch)):
        x1, y1 = keypointListMatch[i][0]+width, keypointListMatch[i][1]
        x2, y2 = keypointListOriginal[i][0], keypointListOriginal[i][1]
        cv2.line(img_tmp, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    cv2.imwrite("result.png",img_tmp)
    
img = cv2.imread("eiffel_tower.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rotated = ndimage.rotate(gray, 4)
v_img = hconcat_resize_min([gray,rotated])
cv2.imwrite("eiffel.png",v_img)
