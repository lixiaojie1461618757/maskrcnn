import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as io
import re

import coco
import utils
import model as modellib
import visualize
from config import Config
#matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/home/lxj/mask_rcnn/Mask_RCNN-master/logs/shapes20181111T1429/mask_rcnn_shapes_0135.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):

    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "/home/lxj/mask_rcnn/Mask_RCNN-master/images/")
#IMAGE_DIR1 = os.path.join(ROOT_DIR, "/home/lxj/mask_rcnn/Mask_RCNN-master/yuantu/")
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM =2560
    IMAGE_MAX_DIM =2560

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (0.05,2,4,8,12)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 500

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
# model.load_weights(COCO_MODEL_PATH, by_name=True)
#model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
#assert model_path != "", "Provide path to trained weights"
#print("Loading weights from ", model_path)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['__background__', 'points','elongated_chains','V_chains','U_chains']

# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
count = os.listdir(IMAGE_DIR)
count95=0
count90=0
count85=0
count70=0
for i in range(0,len(count)):
    path = os.path.join(IMAGE_DIR, count[i])
    if os.path.isfile(path):
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, count[i]))
        #image_pre = skimage.io.imread(os.path.join(IMAGE_DIR1, count[i]))
# Run detection
        #print("file_names",len(file_names))
        results = model.detect([image], verbose=1)
        
# Visualize results
        #ax = get_ax(1)
        r = results[0]
        mask=r['masks']
        class_ids=r['class_ids']
        scores=r['scores']
        file_name=re.sub('.png','', file_names[i])
        mat_path = '/home/lxj/mask_rcnn/Mask_RCNN-master/mask/%3s.mat'%(file_name)
        mat_path1 = '/home/lxj/mask_rcnn/Mask_RCNN-master/class_ids/%3s.mat'%(file_name)
        mat_path2 = '/home/lxj/mask_rcnn/Mask_RCNN-master/scores/%3s.mat'%(file_name)
        io.savemat(mat_path, {'name': r['masks']})
        io.savemat(mat_path1, {'name':r['class_ids']})
        io.savemat(mat_path2, {'name':r['scores']})
        '''
        a=[]
        b=[]
        count1=0
        for i in range(2048):
           for j in range(2048):
               for k in range(50):
                  if(mask[i][j][k]==1):
                      a.append(i)
                      b.append(j)
               count1=count1+1
           len(a) 
           print("len(a)",len(a))
           for i in range(len(a)):
           m[a[i]][b[i]]=1
           print("m",m)

        #class_ids=r['class_ids']
        #scores=r['scores']
        #rois=r['rois']
        #print("mask",mask)
        #print("scores",scores)
        #print("class_ids",class_ids)
        #print("rois",rois)
        #mat_path = '/home/lxj/mask_rcnn/Mask_RCNN-master/'
        #mask_mat = np.empty_like(mask)
        #np.save("mask", mask)
        #np.savetxt("result.txt", scores)
        '''

        visualize.display_instances(count[i],image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
        for i in range(len(r['scores'])):
            if(r['scores'][i]>=0.95):
               count95=count95+1
            if(r['scores'][i]>=0.90):
               count90=count90+1
            if(r['scores'][i]>=0.85):
               count85=count85+1
            if(r['scores'][i]>=0.7):
               count70=count70+1
        print("count95",count95)
        print("count90",count90)
        print("count85",count85)
        print("count70",count70)
        
