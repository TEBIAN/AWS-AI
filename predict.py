
import numpy as np
import pandas as pd

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from torch import optim, cuda
import torch.nn as nn

from PIL import Image
import seaborn as sns



try:
    from torchinfo import summary
    from tqdm.autonotebook import tqdm
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
except:
    # installing torchinfo and then importing
    print("[INFO] Installing torchinfo for network architecture explanation.")
    !pip install torchinfo
    from torchinfo import summary

    # for readability
    print()

    # installing tqdm and then importing
    print("[INFO] Installing tqdm for progress bar.")
    !pip install tqdm
    from tqdm.autonotebook import tqdm

    # for readability
    print()

    # installing torchmetrics and importing MulticlassAccuracy and MulticlassF1Score
    print("[INFO] Installing torchmetrics for computing metrics for training/eval runs.")
    !pip install torchmetrics
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

def load_model(
               model_save_path,
               hidden_layers,
               output_layer,
               optimizer,
               device):
    # building vgg
    model =models.vgg16()
    # swapping classifier head using hidden layers
    model =     nn.Sequential(nn.Linear(224, hidden_layers[0]),
                              nn.ReLU(),
                              nn.Linear(hidden_layers[0], hidden_layers[1]),
                              nn.ReLU(),
                              nn.Dropout(0.4),
                              nn.Linear(hidden_layers[1], output_layer),
                              nn.LogSoftmax(dim=1))

    #
    model.load_state_dict(torch.load(model_save_path, map_location=device),strict=False)
    #model.load_state_dict(torch.load(model_save_path),map_location=torch.device('cpu'))
    model.to(device);
    model.eval();
    # returning model
    return model

# loading model

def load_checkpoint(checkpoint_save_path):

    # returning checkpoint dictionary
    CHECKPOINT_DICT = torch.load(checkpoint_save_path,map_location=torch.device('cpu'))
    return CHECKPOINT_DICT



model_save_path='/content/drive/MyDrive/Colab Notebooks/vgg16.pt'
checkpoint_save_path='/content/drive/MyDrive/Colab Notebooks/VGG16_checkpoint.pth'
file_size = (os.path.getsize(checkpoint_save_path)+os.path.getsize(model_save_path))/(1024 * 1024 * 1024)
print(f'Check point size is :{file_size:.2f} GB')
# loading model and checkpoint
checkpoint_dict = load_checkpoint(checkpoint_save_path)
new_model = load_model(model_save_path,
                       hidden_layers = checkpoint_dict ["hidden_layers"],
                       output_layer = checkpoint_dict ["output_layer"],
                       optimizer = checkpoint_dict ["optimizer"],
                       device ="cpu")

class_to_idx=checkpoint_dict['class_to_idx']
classes=checkpoint_dict['classes']
from PIL import Image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

   '''
    img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224,224)),
    transforms.CenterCrop((224,224)),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
    image = Image.open(image_path)
    img = img_transforms(image)
    return img
def predict(image_path, model, class_to_idx,top_k=5):
    img = process_image(image_path)
    input_batch = img.unsqueeze(0).to(device)

    # Moving model to device and switching to eval mode
    model.to(device)
    model.eval()
    with torch.no_grad():
      output= model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    topk_probs, topk_indices = torch.topk(probabilities, top_k)
    topk_probs_np = topk_probs.numpy(force=True)
    topk_indices_np = topk_indices.numpy(force=True)

    return topk_probs_np, topk_indices_np
probs,classes=predict(img_path, new_model,class_to_idx,5);
