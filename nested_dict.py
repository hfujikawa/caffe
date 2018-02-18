# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 08:28:40 2018
https://stackoverflow.com/questions/21560433/how-to-write-a-nested-dictionary-to-json
@author: hfuji
"""

from __future__ import print_function
import json

out = True

d = {
 "Laptop": {
            "sony": 1,
            "apple": 2,
            "asus": 5,
          },
 "Camera": {
            "sony": 2,
            "sumsung": 1,
            "nikon" : 4,
           },
}

if out:
    with open("my.json","w") as f:
        json.dump(d,f)

samp = {
    "batch_sampler":[ {
      "sampler": {
        "min_scale": 0.3,
        "max_scale": 1.0,
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 2.0,
      },
      "sample_constraint": {
        "min_jaccard_overlap": 0.3
      },
      "max_sample": 1,
      "max_trials": 50
    },
    {
      "sampler": {
        "min_scale": 0.3,
        "max_scale": 1.0,
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 2.0
      },
      "sample_constraint": {
        "min_jaccard_overlap": 0.5
      },
      "max_sample": 1,
      "max_trials": 50
    },
    {
      "sampler": {
        "min_scale": 0.3,
        "max_scale": 1.0,
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 2.0
      },
      "sample_constraint": {
        "min_jaccard_overlap": 0.7
      },
      "max_sample": 1,
      "max_trials": 50
    },
    {
      "sampler": {
        "min_scale": 0.3,
        "max_scale": 1.0,
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 2.0
      },
      "sample_constraint": {
        "min_jaccard_overlap": 0.9
      },
      "max_sample": 1,
      "max_trials": 50
    },
    {
      "sampler": {
        "min_scale": 0.3,
        "max_scale": 1.0,
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 2.0
      },
      "sample_constraint": {
        "max_jaccard_overlap": 1.0
      },
      "max_sample": 1,
      "max_trials": 50
    }]
}

print(samp['batch_sampler'][4])
print('jaccard: ', samp['batch_sampler'][4]['sample_constraint']['max_jaccard_overlap'])

if out:
    with open("samp.json", "w") as f:
        json.dump(samp, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
