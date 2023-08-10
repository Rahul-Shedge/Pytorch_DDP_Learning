import requests
import zipfile
import torch
import yaml
import torch
from torch import nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.utils.utils import read_yaml
import argparse



def data_extraction(config_path):
    configs = read_yaml(config_path)
    # Setup path to data folder
    data_path = Path(configs["data"]["data_folder"])
    image_path = data_path / configs["data"]["image_folder"]

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get(configs["data"]["data_source"])
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...") 
            zip_ref.extractall(image_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--configs","-c",default="configs.yaml")
    parsed_args = args.parse_args()
    data_extraction(parsed_args.configs)

