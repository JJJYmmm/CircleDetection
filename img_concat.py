import cv2
import numpy as np
# img1 = cv2.imdecode(np.fromfile('./picture_source/picture.jpg', dtype=np.uint8), -1)
img1 = cv2.imdecode(np.fromfile('./picture_result/canny_result.jpg', dtype=np.uint8), -1)
img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2 = cv2.imdecode(np.fromfile('./picture_result/hough_result.jpg', dtype=np.uint8), -1)
# 纵向合并
print(img1.shape)
print(img2.shape)
img_total = np.concatenate([img1, img2], axis=1)
cv2.imwrite("./concat_result.jpg", img_total)
cv2.waitKey(0)