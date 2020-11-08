import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from os.path import isfile, join, exists
from os import rename, listdir, rename, makedirs

#bird_specie -> classification
#species -> classes
#specie -> cls

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/lindsey/Documents/Git_Projects/bird_species_classification/")

classes = [
    "fruit",
    "flower",
    "both",
    "ambiguous",
]

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "mask_rcnn/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "train")




class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()


# Create model object in inference mode.# Creat
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# Train Images path
image_path = "/home/lindsey/Documents/Git_Projects/bird_species_classification/train/"

# Load Test images
for classification in classes:
    cls = join(image_path, classification)

    files = listdir(specie)
    # Files sorting
    files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    plants = 1

    if not exists(join("/home/lindsey/Documents/Git_Projects/bird_species_classification/mask_rcnn_crops/", classification)):
    	makedirs(join("/home/lindsey/Documents/Git_Projects/bird_species_classification/mask_rcnn_crops/", classification))
    	
    # Detect birds in each class
    for file in files:

        img_path = join(cls, file)

        image = cv2.imread(img_path, 1)

        result = model.detect([image], verbose=1)

        res = result[0]

        number_of_rois = len(res["rois"])

        imgs = 1

        for j in range(number_of_rois):

            # If potted plant or broccoli is found append then crop the plant/broccoli out of that image
            if res["class_ids"][j] == 51 or res["class_ids"][j] == 59:

                y1, x1, y2, x2 = res["rois"][j]

                crop = image[y1:y2, x1:x2]

                cv2.imwrite(
                    "../mask_rcnn_crops/"
                    + classification
                    + "/"
                    + str(plants)
                    + str(imgs)
                    + ".jpg",
                    crop,
                )

                imgs += 1

        plants += 1
