import numpy as np
import pandas as pd
import pickle
import os
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision import transforms

from dataset import XtoCDataset # our custom dataset class


class ConceptModel(nn.Module):
    def __init__(self):
        super(ConceptModel, self).__init__()
        # Load Inception V3 model with pre-trained weights
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1) # avoid deprecation warning
        # # freeze all layers
        # for param in self.model.parameters():
        #     param.requires_grad = False


        # Modify the top classification layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 312)  # Adjust the number of output classes
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits, aux_logits = self.model(x)
        c = self.sigmoid(logits)
        return c
    

class PredictionModel(nn.Module): # single linear layer logistic reg
    def __init__(self):
        super(PredictionModel, self).__init__()
        self.fc1 = nn.Linear(312, 256)  # Concept vector as input in the first layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 224)
        self.fc3 = nn.Linear(224, 200)  # Output layer for 200 bird species
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, y):
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        y = self.softmax(self.fc3(y))
        #y=self.fc2(y)
        return y
    

class BottleneckModel(nn.Module):
    def __init__(self):
        super(BottleneckModel, self).__init__()
        self.concept_model = ConceptModel()
        self.prediction_model = PredictionModel()

    def forward(self, x):
        concept_probs = self.concept_model(x)    # TODO: need to get output binary? for now its probs
        predictions = self.prediction_model(concept_probs)
        return predictions
    

def load_data(ROOT_DIR):
    # Load datasets from pickle file
    dataset_path = join(ROOT_DIR, 'datasets')
    with open(join(dataset_path, 'train.pkl'), 'rb') as f:
        trainset = pickle.load(f)
    with open(join(dataset_path, 'test.pkl'), 'rb') as f:
        testset = pickle.load(f)
    return trainset, testset

def load_attributes(CUB_DIR):
    with open(join(CUB_DIR, 'attributes.txt'), 'r') as f:
        raw_attributes = f.readlines()
    attributes = np.array([a.split(' ')[1].replace('\n', '') for a in raw_attributes])
    return attributes

def dataset_loader(trainset):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        ])

    dataset = XtoCDataset(trainset, transform=transform)
    return dataset


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    OUT_DIR = join(ROOT_DIR, 'output')
    DATA_DIR = join(ROOT_DIR, 'data')
    CUB_DIR = join(DATA_DIR, 'CUB_200_2011')
    print(f'{ROOT_DIR}\n{OUT_DIR}\n{DATA_DIR}')


    trainset, testset = load_data(ROOT_DIR)
    attributes = load_attributes(CUB_DIR)

    dataset = dataset_loader(trainset[:500])
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BottleneckModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = 500

    # sanity check
    img_batch, class_labels, attr_labels = next(iter(train_loader))
    print(img_batch.shape)

    epoch_losses = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for img_batch, class_labels, attr_labels in train_loader:
            img_batch, class_labels = img_batch.to(device), class_labels.to(device)

            optimizer.zero_grad()  
            outputs = model(img_batch)
            
            loss = criterion(outputs, class_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * img_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')




    # saving model and metrics
    torch.save(model.state_dict(), join(OUT_DIR, 'model.pth'))
    loss_metric = pd.DataFrame({'epoch': [epoch], 'loss': [epoch_losses]})
    loss_metric.to_csv(join(OUT_DIR, 'loss.csv'), index=False)