# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:53:24 2018
https://qiita.com/Alice1017/items/34befe8168cd771f535f
https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
@author: hfuji
"""

#import shutil
import sys
import requests

link = "https://github.com/willyd/caffe-builder/releases/download/v1.1.0/libraries_v120_x64_py27_1.1.0.tar.bz2"
file_name = 'D:\\Develop\\Python\\libraries_v120_x64_py27_1.1.0.tar.bz2'

#res = requests.get(URL, stream=True)
with open(file_name, "wb") as fp:
#    shutil.copyfileobj(res.raw, fp)
    print "Downloading %s" % file_name
    response = requests.get(link, stream=True)
    total_length = response.headers.get('content-length')

    if total_length is None: # no content length header
        fp.write(response.content)
    else:
        dl = 0
        total_length = int(total_length)
        for data in response.iter_content(chunk_size=4096):
            dl += len(data)
            fp.write(data)
            done = int(50 * dl / total_length)
            sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
            sys.stdout.flush()