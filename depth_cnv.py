# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 07:46:17 2017
https://stackoverflow.com/questions/32323922/how-to-convert-a-24-color-bmp-image-to-16-color-bmp-in-python
@author: hfuji
"""

from PIL import Image

iname = 'bin1.png'
oname = 'test.png'

img = Image.open(iname)
newimg = img.convert(mode='P', colors=8)
newimg.save(oname)
