# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 09:54:51 2018
https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
@author: hfuji
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('baboon.jpg', 0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
#mask[50:200, 100:400] = 255
pts = np.array( [ [50,110], [200,140], [300, 400], [110,250] ] )
cv2.fillPoly(mask, [pts], (255,255,255))
masked_img = cv2.bitwise_and(img, img, mask = mask)

cv2.imwrite('masked_img.png', masked_img)
#cv2.imshow('masked img', masked_img)

# calculate histogram with mask
hist_mask = cv2.calcHist([img], [0], mask, [256], [0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_mask)

plt.show()