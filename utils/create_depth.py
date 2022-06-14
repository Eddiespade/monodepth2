import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('../kitti_data/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.jpg', 0)
imgR = cv2.imread('../kitti_data/2011_09_26/2011_09_26_drive_0002_sync/image_03/data/0000000069.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# stereo = cv2.StereoBM_create(numDisparities=64, blockSize=17)

disparity = stereo.compute(imgL, imgR)
cv2.imwrite('depth_map.png', disparity)

disp_v2 = cv2.imread('depth_map.png')
disp_v2 = cv2.applyColorMap(disp_v2, cv2.COLORMAP_JET)

plt.imshow(disp_v2)

cv2.imwrite('depth_map_coloured.png', disp_v2)
plt.show()
