# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 19:53:02 2018

@author: hfuji
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:\\Develop\\Python\\gray256.png', -1)
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
hist = cv2.calcHist([img],[0],None,[256],[0,256])

array = np.asarray(img)
max_val = np.max(array)
min_val = np.min(array)
print(array.shape, min_val, max_val)
data1D = array.flatten()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(data1D, bins=50)

ax.set_xlabel('x')
ax.set_ylabel('freq')
fig.show()