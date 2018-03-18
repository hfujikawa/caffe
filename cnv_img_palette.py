# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 09:15:55 2018
https://stackoverflow.com/questions/29433243/convert-image-to-specific-palette-using-pil-without-dithering
Convert image to specific palette using PIL without dithering
@author: hfuji
"""

import os, sys
import numpy as np
import PIL
from PIL import Image
from VOClabelcolormap import color_palette

def quantizetopalette(silf, palette, dither=False):
    """Convert an RGB or L mode image to use a given P image's palette."""

    silf.load()

    # use palette from reference image made below
    palette.load()
    im = silf.im.convert("P", 0, palette.im)
    # the 0 above means turn OFF dithering making solid colors
    return silf._new(im)

 
row = 10
col = 500
ncolor = 60
array = np.zeros((row*ncolor, col))
for i in range(ncolor):
    array[i*row:i*row+row, :] = 255 / ncolor * i
array = np.uint8(array)
im = Image.fromarray(np.uint8(array))
im.save('gray256.png')
    
files =['gray256.png', 'slice56.png']    

for imgfn in files:
    palettedata = [ 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 255, 255, 255,85,255,85, 255,85,85, 255,255,85] 

#   palettedata = [ 0, 0, 0, 0,170,0, 170,0,0, 170,85,0,] # pallet 0 dark
#   palettedata = [ 0, 0, 0, 85,255,85, 255,85,85, 255,255,85]  # pallet 0 light

#   palettedata = [ 0, 0, 0, 85,255,255, 255,85,255, 255,255,255,]  #pallete 1 light
#   palettedata = [ 0, 0, 0, 0,170,170, 170,0,170, 170,170,170,] #pallete 1 dark
#   palettedata = [ 0,0,170, 0,170,170, 170,0,170, 170,170,170,] #pallete 1 dark sp

#   palettedata = [ 0, 0, 0, 0,170,170, 170,0,0, 170,170,170,] # pallet 3 dark
#   palettedata = [ 0, 0, 0, 85,255,255, 255,85,85, 255,255,255,] # pallet 3 light

#  grey  85,85,85) blue (85,85,255) green (85,255,85) cyan (85,255,255) lightred 255,85,85 magenta (255,85,255)  yellow (255,255,85) 
# black 0, 0, 0,  blue (0,0,170) darkred 170,0,0 green (0,170,0)  cyan (0,170,170)magenta (170,0,170) brown(170,85,0) light grey (170,170,170) 
#  
# below is the meat we make an image and assign it a palette
# after which it's used to quantize the input image, then that is saved 
    palimage = Image.new('P', (16, 16))
    palette256 = color_palette(256, False)
#    palimage.putpalette(palettedata *32)
    palimage.putpalette(palette256)
    oldimage = Image.open(imgfn)
    oldimage = oldimage.convert("RGB")
    newimage = quantizetopalette(oldimage, palimage, dither=False)
    dirname, filename= os.path.split(imgfn)
    name, ext= os.path.splitext(filename)
    newpathname= os.path.join(dirname, "cga-%s.png" % name)
    newimage.save(newpathname)

#   palimage.putpalette(palettedata *64)  64 times 4 colors on the 256 index 4 times, == 256 colors, we made a 256 color pallet.