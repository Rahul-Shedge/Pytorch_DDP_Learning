import os
import torch
# import TinyVGG, create_dataloaders,train
from src.utils import utils
from datetime import datetime
from src.data_preprocessing import create_dataloaders
from src.model_creation import TinyVGG
from src.training_steps import train
import argparse
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp 




def ddp_setup(rank,world_size):
    """
    Args:

    """
    os.envrion["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl",rank=rank,world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank,world_size,config_path,target_dir,model_name):
    configs = utils.read_yaml(config_path)
    world_size= world_size
    train_dir = os.path.join(configs["data"]["data_folder"],os.path.join(configs["data"]["image_folder"],"train"))
    test_dir = os.path.join(configs["data"]["data_folder"],os.path.join(configs["data"]["image_folder"],"test"))
    batch_size = configs["config"]["BATCH_SIZE"]
    learning_rate = configs["config"]["learning_rate"]
    epochs = configs["config"]["epochs"]
    # device = configs["config"]["device"]
    rank = rank
    # num_workers = configs["config"]["NUM_WORKERS"]


    ddp_setup(rank=rank,world_size=world_size)

    train_loader,test_loader, classes = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=batch_size,
    #    transform = defined in the internal code only we can redefined outside here if neeeded
        # num_workers=num_workers
    )


    model = TinyVGG(
        input_shape = configs["config"]["in_channels"],
        hidden_units = configs["config"]["hidden_units"],
        output_shape = len(classes),

    ).to(rank)


    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate
        )

    train(
        model= model,
        train_dataloader= train_loader,
        test_dataloader = test_loader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs=epochs,
        # device = device,
        rank= rank
    )

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir = target_dir,
                    model_name = str(model_name) +str(".pth"))


if __name__=="__main__":
    dt = datetime.now()
    uniqueness = str(dt).replace(" ","_").replace(":","__").split(".")[0]
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="configs.yaml")
    args.add_argument("--target_dir","-t",default="models")
    args.add_argument("--model_name","-m",default="Tiny_VGG_model"+str(uniqueness))
    parsed_args = args.parse_args()
    world_size = torch.cuda.device_count()
    try:
        mp.spawn(main,args=(world_size,parsed_args.config,parsed_args.target_dir,parsed_args.model_name),nprocs=world_size)
    except Exception as e:
        raise e
    
