# Import necessary packages.
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

### Parameter
dataset_path = "./images"

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 1

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((512, 512)),
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
dataset = ImageFolder(dataset_path, transform=test_tfm)
targets = dataset.targets
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = torch.load("220126_3_inceptionv2_resnet_model.pt", map_location='cpu')
model.device = device

# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
model.eval()

# Initialize a list to store the predictions.
predictions = []
# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch
    # np_arr = imgs.cpu().detach().numpy()
    # print('a', np_arr.shape)
    # np_arr = np_arr.squeeze()
    # np_arr = np.moveaxis(np_arr, 0, -1)
    # print('b', np_arr.shape)
    # print(type(np_arr))
    # cv2.imshow("windows", np_arr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # groundtruth += labels.tolist()
    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        # print(imgs.to(device))
        # print("input shape:", imgs.to(device).shape)
        logits = model(imgs.to(device))
        print(type(logits))
        print("output shape:", logits.shape)

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
print(predictions)
for i in range(len(predictions)):
    if predictions[i] == 0:
        print("428")
    elif predictions[i] == 1:
        print("n_428")