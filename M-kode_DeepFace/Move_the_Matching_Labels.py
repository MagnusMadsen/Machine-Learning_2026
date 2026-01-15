import os
import time
import uuid
import cv2

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt

for folder in ['TRAIN','TEST','VALIDATION']:
    for file in os.listdir(os.path.join('Data', folder, 'Images')):
        
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('Data','Labels', filename)
        if os.path.exists(existing_filepath): 
            new_filepath = os.path.join('Data',folder,'Labels',filename)
            os.replace(existing_filepath, new_filepath)