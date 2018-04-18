# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:28:50 2018

@author: hfuji
"""

import numpy as np
import cv2

img_org = cv2.imread("fish-bike_Final.tif", 0)
height, width = img_org.shape[0:2]

xunit = int(width / 3)
sign_x = 1
if sign_x > 0:
    shift_x = int(xunit*2 - width/2 + np.random.randint(xunit))
else:
    shift_x = int(xunit - width/2 - np.random.randint(xunit))

img_shift = np.zeros((height,width), np.uint8)
mean_val = np.mean(img_org)
img_shift.fill(mean_val)

if shift_x > 0:
    xs = shift_x
    shift_w = width - xs
    img_shift[:, xs:xs+shift_w] = img_org[:, :shift_w]
if shift_x < 0:
    xs = -shift_x
    shift_w = width - xs
    img_shift[:, :shift_w] = img_org[:, xs:xs+shift_w]
cv2.imwrite("img_shift.png", img_shift)