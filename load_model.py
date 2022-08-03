from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

import time

from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import *

from PIL import Image
from pathlib import Path

import streamlit as st


class ConvNets(nn.Module):
    def __init__(self):
        super(ConvNets, self).__init__()
        # convolution layer (sees 224 * 224 * 3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # convolution layer (sees 112 * 112 * 16 image tensor)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # convolution layer (sees 56 * 56 * 32 image tensor)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Relu activation function
        self.relu = nn.ReLU()
        # Maxpool function to reduce the kernel size
        self.maxpool = nn.MaxPool2d(2)
        # Linear layer (sees 28 * 28 * 64 tensors)
        self.fc = nn.Linear(28 * 28 * 64, 102)
        # Final out layer (see 102 tensors and return prediction 1)
        self.output = nn.Linear(102,1)
        # sigmoid activation function for final prediction
        self.sigmoid = nn.Sigmoid()
        # dropout function to reduce overfitting
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # first convolution layer
        out = self.maxpool(self.relu(self.conv1(x)))
        # Second convoluton layer
        out = self.maxpool(self.relu(self.conv2(out)))
        # Third convolution layer
        out = self.maxpool(self.relu(self.conv3(out)))
        # flattening the output
        out = out.view(-1, 28 * 28 * 64)
        # adding dropout to the model
        out = self.dropout(out)
        # first fully connected layer
        out = self.relu(self.fc(out))
        # adding dropout to the model
        out = self.dropout(out)
        # final fully connected layer with sigmoid function
        out = self.sigmoid(self.output(out))
        return out

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(BytesIO(image_data))
    else:
        return None


location = Path(Path(__file__).parents[0], "models", "best_model.bin")
print(location)


def load_model(location):
    model = ConvNets()
    state_dict = torch.load(location.__str__())
    model.load_state_dict(state_dict)
    return model


def predict(model, image):
    preprocess = transforms.Compose([
                        transforms.RandomHorizontalFlip(), 
                        transforms.RandomRotation(25),    
                        transforms.Resize([224, 224]),    
                        transforms.ToTensor(),           
                        transforms.Normalize([0.3692, 0.5223, 0.2339], [0.1710, 0.1216, 0.1737]),
                               ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    result = output.round()

    if result == 1:
        st.write("This maize plant has been infested by armyworms and needs urgent pesticide")
    else:
        st.write("This plant is perfectly healthy")

def main():
    st.title('Diseases detection in plants using ML demo')
    model = load_model(location)
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, image)
        # time.sleep(5)
        # st.write("This maize plant has been infested by armyworms and needs urgent pesticide")


if __name__ == "__main__":
    main()