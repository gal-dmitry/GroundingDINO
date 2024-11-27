# python dino2yolo_dataset.py --config_path configs/catdog_train_01.yml


import shutil
import argparse
from src.gdino_predictor import DinoPredictor, load_args


if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument('--config_path', type=str, required=True)
   args = parser.parse_args()

   config = load_args(args.config_path)
   predictor_kwargs = config["PREDICTOR_KWARGS"]
   print(predictor_kwargs)

   # 1. predict
   predictor = DinoPredictor(**predictor_kwargs)
   predictor()

   # 2. add config
   dir_name = config['ROOT']
   output_dir = f"{'/'.join(dir_name.split('/')[:-2])}/_archives"
   output_filename = f"{output_dir}/yolo_{config['CHUNK']}"

   src_cfg = "/home/ubuntu/DMITRII/GroundingDINO/configs/data.yaml"
   res_cfg = f"{dir_name}/data.yaml"
   shutil.copy(src_cfg, res_cfg)

   # 3. archive
   shutil.make_archive(output_filename, 'zip', dir_name)
