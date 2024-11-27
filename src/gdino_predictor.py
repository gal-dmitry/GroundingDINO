
import sys
import os.path as op
sys.path.append(
    op.abspath(op.join(__file__, op.pardir))
)

import torch
import random
import numpy as np

import os
import cv2
import supervision as svn
from typing import List
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from groundingdino.util.inference import Model


"""
Hyperparameters
"""
CLASSES = ['cat', 'dog']
GDINO_ARGS_PATH = "/home/ubuntu/DMITRII/EmbleMLDev3.0/GroundingDINO/src/gdino_config_01.yaml"


"""
Seed
"""
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


"""
Utils
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


def get_dirs(root):
   split='train'
   img_dir = f"{root}/images/{split}"
   labels_dir = f"{root}/labels/{split}"
   yaml_path = f"{root}/data.yaml"
   return img_dir, labels_dir, yaml_path


# def get_model_kwargs(model_kwargs):
#    prefix = os.getcwd() + '/'
#    model_kwargs["model_config_path"] = model_kwargs["model_config_path"].replace(prefix, '')
#    model_kwargs["model_checkpoint_path"] = model_kwargs["model_checkpoint_path"].replace(prefix, '')
#    print(model_kwargs)
#    return model_kwargs


def get_gdino_kwargs(
   root, 
   gdino_args_path="/home/ubuntu/DMITRII/EmbleMLDev3.0/GroundingDINO/src/gdino_config_01.yaml",
):
   args = load_args(gdino_args_path)
   # model_kwargs = get_model_kwargs(args['MODEL_KWARGS'])
   model_kwargs = args['MODEL_KWARGS']
   predict_kwargs = args['PREDICT_KWARGS']
   dataset_kwargs = args['DATASET_KWARGS']

   img_dir, labels_dir, _ = get_dirs(root)
   dataset_kwargs['annotations_directory_path'] = labels_dir
   read_kwargs = dict(
      directory=img_dir,
      extensions=args['IMG_EXT'],
   )

   return model_kwargs, read_kwargs, predict_kwargs, dataset_kwargs



"""
Predictor
"""
class DinoPredictor:
   """
   https://www.kdnuggets.com/2023/05/automatic-image-labeling-grounding-dino.html
   """
   def __init__(
      self,
      root,
      classes=CLASSES,
      gdino_args_path=GDINO_ARGS_PATH,
   ): 
      
      model_kwargs, read_kwargs, predict_kwargs, dataset_kwargs = get_gdino_kwargs(root, gdino_args_path)
      self.model = Model(**model_kwargs)
      self.image_paths = svn.list_files_with_extensions(**read_kwargs)
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