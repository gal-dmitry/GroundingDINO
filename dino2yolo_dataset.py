# python dino2yolo_dataset.py --gdino_args_path configs/<config>


import shutil
import argparse
from src.gdino_predictor import load_args, DinoPredictor
TO_CVAT = '_to_cvat'


if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument('--config_path', type=str, required=True)
   parser.add_argument(
      '--src_yolo_cfg', 
      type=str, 
      default="/home/ubuntu/DMITRII/EmbleMLDev3.0/GroundingDINO/src/data.yaml"
   )
   args = parser.parse_args()

   config = load_args(args.config_path)
   print(config)
   
   predictor_kwargs = config["PREDICTOR_KWARGS"]

   # 1. predict labels
   predictor = DinoPredictor(**predictor_kwargs)
   predictor()

   # 2. copy yolo config & archive yolo root
   root, chunk = config['ROOT'], config['CHUNK']
   zip_dir = f"{'/'.join(root.split('/')[:-2])}/{TO_CVAT}"
   zip_name = f"{zip_dir}/yolo_{chunk}"

   res_yolo_cfg = f"{root}/data.yaml"
   shutil.copy(args.src_yolo_cfg, res_yolo_cfg)

   shutil.make_archive(zip_name, 'zip', root)
