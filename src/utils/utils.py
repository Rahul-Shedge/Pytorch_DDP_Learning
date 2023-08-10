
import pandas as pd
import yaml
import os
import logging
import torch



def read_yaml(path):
    with open(path,"r") as yaml_file:
        content = yaml.safe_load(yaml_file)
        # print("**"*10)
        # print(content)
    return content

def creat_dir(path_to_dir:list)->None:
    for path in path_to_dir:
        os.makedirs(path,exist_ok=True)


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    print(target_dir)
    target_dir_path = target_dir
    # target_dir_path.mkdir(parents=True,
    #                     exist_ok=True)
    os.makedirs(target_dir_path,exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = os.path.join(target_dir_path, model_name)

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.module.state_dict(),
             f=model_save_path)