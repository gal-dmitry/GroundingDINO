python dino2yolo_dataset.py --config_path configs/01_train.yml
# python dino2yolo_dataset.py --config_path configs/01_val.yml

# https://www.kdnuggets.com/2023/05/automatic-image-labeling-grounding-dino.html


import torch
import random
import numpy as np

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

import argparse
from hyperpyyaml import load_hyperpyyaml

import os
import cv2
import supervision as svn
from typing import List
from tqdm import tqdm
from groundingdino.util.inference import Model



"""
Hyperparameters
"""
# model kwargs
GROUNDING_DINO_CONFIG_PATH = os.path.join("groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("weights/groundingdino_swint_ogc.pth")
MODEL_KWARGS=dict(
   model_config_path=GROUNDING_DINO_CONFIG_PATH, 
   model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
)

# classes
CLASSES = ['cat','dog'] #add the class name to be labeled automatically
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.15
PREDICT_KWARGS=dict(
   box_threshold=BOX_TRESHOLD,
   text_threshold=TEXT_TRESHOLD,
)

# data kwargs
MIN_IMAGE_AREA_PERCENTAGE = 0.0
MAX_IMAGE_AREA_PERCENTAGE = 0.1
APPROXIMATION_PERCENTAGE = 0.0

ROOT = "datasets/dino2yolo"
SPLIT = "train"
IMG_DIR = f"{ROOT}/images/{SPLIT}"
LABELS_DIR = f"{ROOT}/labels/{SPLIT}"
YAML_PATH = f"{ROOT}/data.yaml"
IMG_EXT = ['jpg', 'jpeg', 'png']

READ_KWARGS=dict(
   directory=IMG_DIR,
   extensions=IMG_EXT,
)

DATASET_KWARGS=dict(
   images_directory_path=None,
   annotations_directory_path=LABELS_DIR,
   data_yaml_path=None,
   min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
   max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
   approximation_percentage=APPROXIMATION_PERCENTAGE,
)



"""
Predictor
"""
def load_args(config_name):
    with open(config_name) as file:
        args = load_hyperpyyaml(file)
    return args


def enhance_class_name(class_names: List[str]) -> List[str]:
   return [
       f"all {class_name}s"
       for class_name
       in class_names
   ]


class DinoPredictor:

   def __init__(
      self,
      classes=CLASSES,
      model_kwargs=MODEL_KWARGS,
      predict_kwargs=PREDICT_KWARGS,
      read_kwargs=READ_KWARGS,
      dataset_kwargs=DATASET_KWARGS,
   ):
      self.model = Model(**model_kwargs)
      self.image_paths = svn.list_files_with_extensions(**read_kwargs)
      print(self.image_paths)
      assert len(self.image_paths) > 0
      self.classes = classes
      self.predict_kwargs = predict_kwargs
      self.dataset_kwargs = dataset_kwargs


   def predict_dataset(self):

      images = {}
      annotations = {}

      for image_path in tqdm(self.image_paths):
         image_name = image_path.name
         image_path = str(image_path)
         image = cv2.imread(image_path)

         detections = self.model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=self.classes),
            **self.predict_kwargs,
         )
         detections = detections[detections.class_id != None]
         images[image_name] = image
         annotations[image_name] = detections

      return images, annotations
   

   def save_dataset(
      self,
      images,
      annotations,
   ):
      svn.DetectionDataset(
         classes=self.classes,
         images=images,
         annotations=annotations,
      ).as_yolo(**self.dataset_kwargs)


   def __call__(self):
      images, annotations = self.predict_dataset()
      self.save_dataset(images, annotations)
      print("done !")



if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument('--config_path', type=str, required=True)
   args = parser.parse_args()

   config = load_args(args.config_path)
   predictor_kwargs = config["PREDICTOR_KWARGS"]
   print(predictor_kwargs)
   predictor = DinoPredictor(**predictor_kwargs)
   predictor()
