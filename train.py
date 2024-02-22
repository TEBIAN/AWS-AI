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
data_dir = '/content/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(),
    transforms.Resize((224,224)),
    transforms.CenterCrop((224,224)),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224,224),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.CenterCrop((224,224)),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir,transform=train_transforms)
valid_dataset= datasets.ImageFolder(root=valid_dir, transform=valid_transforms)
test_dataset= datasets.ImageFolder(root=test_dir,transform=test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
valid_dataloader= torch.utils.data.DataLoader(valid_dataset,batch_size=64,shuffle=True)
test_dataloader= torch.utils.data.DataLoader(test_dataset,batch_size=64, shuffle=True)

#define hyperparameters
CLASSES=train_dataset.classes
num_classes=len(CLASSES)
class_to_idx=train_dataset.class_to_idx
idx_to_class = {val: key for (key, val) in class_to_idx.items()}
hidden_layer_1=256
hidden_layer_2=128
# TODO: Build and train your network
# done
model = models.vgg16(pretrained=True)
# Freeze parameters so we don't backprop through them
#done
for param in model.parameters():
    param.requires_grad = False

#Define network architecture
n_inputs = model.classifier[6].in_features

# n_inputs will be 4096 for this case
# Add on classifier
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, hidden_layer_1),
    nn.ReLU(),
    nn.Linear(hidden_layer_1, hidden_layer_2),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(hidden_layer_2, num_classes),
    nn.LogSoftmax(dim=1))
#define the criterion and optimizer
#done
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# move the model to GPU
device=torch.device("cuda:0" if torch.cuda.is_available else "cpu")

print(device)
model=model.to(device)


    model.train()
    ii = 0

    for data, label in train_dataloader:
      ii += 1
      data, label = data.to(device), label.to(device)
      optimizer.zero_grad()
      output = model(data)

      loss = criterion(output, label)
      loss.backward()
      optimizer.step()

      # Track train loss by multiplying average loss by number of examples in batch
      train_loss += loss.item() * data.size(0)

      # Calculate accuracy by finding max log probability
      _, pred = torch.max(output, dim=1) # first output gives the max value in the row(not what we want), second output gives index of the highest val
      correct_tensor = pred.eq(label.data.view_as(pred)) # using the index of the predicted outcome above, torch.eq() will check prediction index against label index to see if prediction is correct(returns 1 if correct, 0 if not)
      accuracy = torch.mean(correct_tensor.type(torch.FloatTensor)) #tensor must be float to calc average
      train_acc += accuracy.item() * data.size(0)
      if ii%15 == 0:
        print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_dataloader):.2f}% complete.')

    model.epochs += 1
    with torch.no_grad():
      model.eval()

      for data, label in valid_dataloader:
        data, label = data.to(device), label.to(device)

        output = model(data)
        loss = criterion(output, label)
        valid_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(label.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        valid_acc += accuracy.item() * data.size(0)

      train_loss = train_loss / len(train_dataloader.dataset)
      valid_loss = valid_loss / len(valid_dataloader.dataset)

      train_acc = train_acc / len(train_dataloader.dataset)
      valid_acc = valid_acc / len(valid_dataloader.dataset)

      history.append([train_loss, valid_loss, train_acc, valid_acc])

      if (epoch + 1) % print_every == 0:
        print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tTraining Accuracy: {100 * train_acc:.2f}%')
        print(f'\t\tValidation Loss: {valid_loss:.4f}\t Validation Accuracy: {100 * valid_acc:.2f}%')

      if valid_loss < valid_loss_min:
        torch.save(model.state_dict(), save_location)
        stop_count = 0
        valid_loss_min = valid_loss
        valid_best_acc = valid_acc
        best_epoch = epoch

      else:
        stop_count += 1

        # Below is the case where we handle the early stop case
        if stop_count >= early_stop:
          print(f'\nEarly Stopping Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
          model.load_state_dict(torch.load(save_location))
          model.optimizer = optimizer
          history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc','valid_acc'])
          return model, history

  model.optimizer = optimizer
  print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')

  history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
  return model, history,best_epoch

model, history,best_epoch = train(
    model,
    criterion,
    optimizer,
    train_dataloader,
    valid_dataloader,
    save_location='/content/images.pt',
    early_stop=7,
    n_epochs=16,
    print_every=2)
def test(model, test_loader, criterion):
  with torch.no_grad():
    model.eval()
    test_acc = 0
    for data, label in test_loader:
      data, label = data.to(device), label.to(device)

      output = model(data)

      _, pred = torch.max(output, dim=1)
      correct_tensor = pred.eq(label.data.view_as(pred))
      accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
      test_acc += accuracy.item() * data.size(0)

    test_acc = test_acc / len(test_loader.dataset)
    return test_acc

test_acc = test(model.cuda(), test_dataloader, criterion)
print(f'The model has achieved an accuracy of {100 * test_acc:.2f}% on the test dataset')

# Saving modeltorch.save(model.state_dict(), PATH)
def save_model(model,
               model_name):
    save_path = f"./{model_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] {model_name} saved to {save_path}")
    return save_path

#save checkpoint
def save_checkpoint(optimizer,
                    criterion,
                    epoch,
                    device_trained_on,
                    hidden_layers,
                    output_layer,
                    classes,
                    class_to_idx,
                    checkpoint_name="checkpoint"):


    save_path = f"./{checkpoint_name}.pth"
    torch.save({"optimizer": optimizer.state_dict(),
                "criterion": criterion,
                "epoch": epoch,
                "device_trained_on": device_trained_on,
                "hidden_layers": hidden_layers,
                "output_layer": output_layer,
                "classes": classes,
                "class_to_idx": class_to_idx}, save_path)
    print(f"[INFO] The general checkpoint has been saved to: {save_path}")
    return save_path

# saving model and checkpoint
model_save_path = save_model(model, "vgg16")
checkpoint_save_path = save_checkpoint(optimizer,
                                  criterion,
                                  epoch = best_epoch,
                                  device_trained_on = device,
                                  hidden_layers = (hidden_layer_1, hidden_layer_2),
                                  output_layer = num_classes,
                                  classes = CLASSES,
                                  class_to_idx = class_to_idx,
                                  checkpoint_name = "VGG16_checkpoint")