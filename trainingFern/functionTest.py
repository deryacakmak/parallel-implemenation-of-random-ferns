"""
###### Test Keypoint Detection  ######

image = readImage("eiffel_tower.png")
image = applySmoothing(addNoise(image))
keypoints = detectKeypoint(image)
for i in keypoints:
    image[i[0]][i[1]] = 255
    
cv2.imwrite("keypointDetection.png",image)

"""


"""
###### Test Apply Affine Deformation  ######

image = readImage("eiffel_tower.png")
image = applySmoothing(addNoise(image))
warp_dst, newKeypoints = applyAffineDeformation(image, keypoints.tolist())
cv2.imwrite("applyAffineDeformation.png",warp_dst)

"""


"""
###### Test Find Coordinate to Modified Image  ######

image = readImage("eiffel_tower.png")
keypoints = detectKeypoint(image)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
warp_dst, newKeypoints = applyAffineDeformation(image, keypoints.tolist())
for i in newKeypoints:
    warp_dst[i[1][0]][i[1][1]] = 255

"""

"""

###### Concanate Image  ######

img = cv2.imread("eiffel_tower.png")
img2 = cv2.imread("rotatedImage.png")
im3 = hconcat_resize_min([img,img2])

cv2.imwrite("concatanate.png",im3)

"""

