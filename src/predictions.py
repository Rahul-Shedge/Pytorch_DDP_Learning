import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List,Tuple
import argparse
from PIL import Image
from src.model_creation import TinyVGG
from src.data_transforms import torchvision_transform_test
from src.utils.utils import read_yaml



device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
            model_path:str,
            input_shape:int,
            hidden_units:int,
            output_shape:int
            ):
    model = TinyVGG(input_shape,
    hidden_units,
    output_shape
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    return model


def predict_and_plot(
    model:torch.nn.Module,
    class_names:str,
    image_path:str,
    image_size:Tuple[int,int]= (224,224),
    transforms:torchvision.transforms = None,
    device: torch.device=device
):

    img = Image.open(image_path)


    if transforms is not None:
        image_transform = transforms
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
            # ,   transforms.Normalize(
            #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            #     ),

        ])
    
    model.to(device)

    model.eval()

    with torch.inference_mode():

        transformed_img=image_transform(img).unsqueeze(dim = 0)

        target_image_pred = model(transformed_img.to(device))

    target_img_prob = torch.softmax(target_image_pred,dim=1)
    target_image_label = torch.argmax(target_img_prob,dim=1)

    plt.figure()
    
    plt.imshow(img)
    plt.title(
        "Pred : "+ str(class_names[target_image_label.item()]) \
        + " | "+ "Prob: "+str(target_img_prob.max().item())
    )
    # plt.ion()
    plt.axis(False)
    plt.show(block=False)
    plt.pause(10)
    plt.close() 


if __name__ =="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path","-m",default="./models/Tiny_VGG_model2023-07-30_21__16__44.pth")
    args.add_argument("--image_path","-i",default="./Images/175783.jpg")
    args.add_argument("--configs","-c",default="configs.yaml")
    parsed_args = args.parse_args()

    try:
        configs = parsed_args.configs
        configs = read_yaml(configs)

        class_names = configs["data"]["class_names"]
        input_shape = configs["config"]["in_channels"],
        # print(input_shape[0])
        hidden_units = configs["config"]["hidden_units"],
        # print(hidden_units)
        output_shape = len(class_names),


        # print(class_names)
        model_path = parsed_args.model_path
        image_path = parsed_args.image_path

        newmodel = load_model(model_path=model_path,
            input_shape=input_shape[0],
            hidden_units=hidden_units[0],
            output_shape=output_shape[0])

        predict_and_plot(
            model=newmodel,
            class_names=class_names,
            image_path=image_path,
            transforms=torchvision_transform_test(),
            device="cpu"
        )
        print("Run successfully.")
    except Exception as e:
        raise e






