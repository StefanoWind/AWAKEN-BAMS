# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:44:25 2025

@author: sletizia
"""

from PIL import Image

source=input('Figure:')
img = Image.open(source+'.png')
img.convert("RGB").save(source+'.pdf')