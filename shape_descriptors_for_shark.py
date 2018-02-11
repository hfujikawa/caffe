# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 12:53:29 2018
https://stackoverflow.com/questions/46395836/shape-descriptors-for-shark-in-water-silhouette-to-remove-false-positives
@author: hfuji
"""

import numpy as np
import os, cv2

dir_pathIN = 'D:\\Develop\\Python'

FILES=os.listdir(dir_pathIN)


#HSV thresholds
channel1Min = 0
channel1Max = 179
channel2Min = 0
channel2Max = 123
channel3Min = 0
channel3Max = 134

AreaMin=200
AreaMax=1100 
aspect_ratio_min = 0.6
aspect_ratio_max = 1.5
eccentricity_min=0.7
eccentricity_max=0.95
solidity_min=0.65
ComplexityPerim_max=40

for file in FILES: 
    if file.endswith('.jpg') or file.endswith('.JPG'):
        filepath = os.path.join(dir_pathIN, file)
        filename, file_extension = os.path.splitext(filepath)
    
        im = cv2.imread(filepath)
        im_copy = im
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) 
        lower_blue = np.array([channel1Min, channel2Min, channel3Min])
        upper_blue = np.array([channel1Max, channel2Max, channel3Max])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(im, im, mask=mask)
        ret, thresh = cv2.threshold(mask, 125, 255, 0)
#        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        AY = len(im)
        AX = len(im[0])
    
        contours_NOT_fish = []  
        contours_final = []
        contours_area=[]
        contours_AR = []
    
        for con in contours:
            area = cv2.contourArea(con)
            if (AreaMin < area < AreaMax):
                contours_area.append(con)
            else:
                contours_NOT_fish.append(con)
    
        for con in contours_area: # ASPECT RATIO
            rect = cv2.minAreaRect(con)
            center = rect[0]
            size = rect[1]
            angle = rect[2]
            w = size[0]
            h = size[1]
            if h == 0:
                break
            aspect_ratio_rot_box = float(w) / h
            if aspect_ratio_rot_box > aspect_ratio_min or aspect_ratio_rot_box < aspect_ratio_max:
                contours_AR.append(con)
            else:
                contours_NOT_fish.append(con)
    
        for con in contours_AR: #Solidity,eccentricity,Perimeter complexity
            M = cv2.moments(con)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Solidity
            area = cv2.contourArea(con)
            hull = cv2.convexHull(con)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
    
            # eccentricity
            ellipse = cv2.fitEllipse(con)
            # center, axis_length and orientation of ellipse
            (center, axes, orientation) = ellipse
            # length of MAJOR and minor axis
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            eccentricity = np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)
    
            #Perimeter complexity
            perimeter = cv2.arcLength(con, True)
    
            if area == 0:
                ComplexityPerim=0
                continue
            else:
                ComplexityPerim= perimeter * perimeter/area
    
            if eccentricity_min <= eccentricity <= eccentricity_max and solidity >= solidity_min and ComplexityPerim<=ComplexityPerim_max:
                contours_final.append(con)
            else:
                contours_NOT_fish.append(con)
    
        cv2.drawContours(im_copy, contours_final, -1, (0, 0, 255), 1)  # Draw with All contours
        cv2.imwrite(filename + '_Final.tif', im_copy)

