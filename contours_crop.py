# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:46:42 2017

@author: hfuji
"""

import cv2
import numpy as np
from PIL import Image

height = 300
width = 300
img = np.zeros([height, width, 3], dtype=np.uint8)
pts = np.array( [ [10,10], [10,100], [100, 100], [100,10] ] )
cv2.fillPoly(img, [pts], (255,255,255))
pts = np.array( [ [110,110], [130,140], [200, 200], [110,150] ] )
cv2.fillPoly(img, [pts], (255,255,255))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
cv2.imwrite('bin.png', thresh)

#contours, _= cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
img = cv2.drawContours(img, [cnt], -1, (0,255,0), -1)
cv2.imwrite('cnt_img.png', img)

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
for i in range(len(contours)):
    contour = contours[i]
    print i, len(contour)
    cnt_lst = []
    for xy in contour:
        print i, xy
        cnt_lst.append((xy[0][0], xy[0][1]))
    cnt_arr = np.array(cnt_lst)
    img_blk = np.zeros([height, width, 3], dtype=np.uint8)
    cv2.fillPoly(img_blk, [cnt_arr], (255,255,255))
    img_poly = img_blk.copy()
    temp = cv2.dilate(img_blk,element)
    temp = cv2.subtract(temp,img_poly)
    cv2.fillPoly(temp, [cnt_arr], (0,0,255))
#    skel = cv2.bitwise_or(skel,temp)
    tempRGB = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    img24Color = Image.fromarray(tempRGB)
    img8Color = img24Color.convert(mode='P', colors=8)
    img8Color.save('img8Color.png')
    cv2.imwrite('bin{}.png'.format(i), temp)
    cv2.imshow('poly image', temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
