# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 20:48:41 2018
https://stackoverflow.com/questions/15229951/opencv-locations-of-all-non-zero-pixels-in-binary-image
@author: hfuji
"""

import numpy as np
import cv2

img_gray = cv2.imread('bin0.png', 0)
ret, thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
bin, contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
mask = np.zeros(img_gray.shape,np.uint8)
height, width = img_gray.shape
margin = 5
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    x,y,w,h = cv2.boundingRect(contours[i])
    print('x,y,w,h: ', x,y,w,h)
    perim = cv2.arcLength(contours[i], True)
    cv2.drawContours(mask,[contours[i]],0,255,-1)
    cv2.imwrite('mask{}.png'.format(i), mask)
    pixelpoints_a = np.transpose(np.nonzero(mask))
    roi = mask[margin:height-margin, margin:width-margin]
    pixelpoints_b = cv2.findNonZero(roi)
    print('area, perim, a, b: ', area, perim, pixelpoints_a.shape, pixelpoints_b.shape)