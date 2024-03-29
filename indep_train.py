import os
from os import listdir
from os.path import isfile, isdir, join
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,SubsetRandomSampler
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import ToTensor, Resize
from torchvision.datasets import ImageFolder
from collections import defaultdict as ddict
import random
import numpy as np

class ConceptModel(nn.Module):
    def __init__(self):
        super(ConceptModel, self).__init__()
        # Load Inception V3 model with pre-trained weights
        self.model = models.inception_v3(pretrained=True)

         # # freeze all layers
        #for param in self.model.parameters():
        #    param.requires_grad = False
        
        # Modify the top classification layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 312)  # Adjust the number of output classes

    def forward(self, x):
        return self.model(x)


class PredictionModel(nn.Module): # single linear layer logistic reg
    def __init__(self):
        super(PredictionModel, self).__init__()
        self.fc1 = nn.Linear(312, 256)  # Concept vector as input in the first layer
        self.fc2 = nn.Linear(256, 224)
        self.fc3 = nn.Linear(224, 200)  # Output layer for 200 bird species
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, c):
        c = self.relu(self.fc1(c))
        c = self.relu(self.fc2(c))
        c=self.fc3(c)
        #c = self.softmax(self.fc3(c))
        return c

# Define your training loop here
def train_c_model(model, train_loader, criterion, optimizer, num_epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch['img'][0]
            labels = batch['attribute_label']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs,_ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Define your training loop here
def train_p_model(model, train_loader, criterion, optimizer, num_epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch['attribute_label']
            labels = batch['class_label']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            predicted = torch.round(outputs)
            classes = torch.argmax(predicted, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Calculate accuracy function
def calculate_accuracy(model_c, model_p, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_p.eval()
    model_c.eval()

    total_p = 0
    total_c = 0
    correct_p = 0
    accuracy_c = 0
    with torch.no_grad():
        accuracy = 0
        for batch in test_loader:
            inputs = batch['img'][0]
            concepts = batch['attribute_label']
            labels = batch['class_label']
            inputs, concepts, labels = inputs.to(device), concepts.to(device), labels.to(device)
            conc = model_c(inputs)
            outputs_1 = model_p(conc)
            #outputs_2 = model_p(concepts)
            
            accuracy_c += binary_accuracy(conc, concepts)
            total_c+=1

            predicted = torch.round(outputs_1)
            classes = torch.argmax(predicted, dim=1)

            #print(classes)
            #print(labels)

            correct_p += (classes == labels).sum().item()
            total_p += labels.size(0)
    accuracy_c /= total_c
    accuracy_p = correct_p/total_p
    return accuracy_c.detach().numpy(),accuracy_p

def binary_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions
    output and target are Torch tensors
    """
    pred = output.cpu() >= 0.5
    acc = (pred.int()).eq(target.int()).sum()
    acc = acc*100 / np.prod(np.array(target.size()))
    return acc

def extract_data(data_dir,dataset):
    cwd = os.getcwd()
    data_path = join(cwd, data_dir + "\\images")
    val_ratio = 0.1

    path_to_id_map = dict()  # map from full image path to image id
    with open(data_path.replace("images", "images.txt"), "r") as f:
        for line in f:
            items = line.strip().split()
            path_to_id_map[join(data_path, items[1])] = int(items[0])

    attribute_labels_all = ddict(
        list
    )  # map from image id to a list of attribute labels
    attribute_certainties_all = ddict(
        list
    )  # map from image id to a list of attribute certainties
    attribute_uncertain_labels_all = ddict(
        list
    )  # map from image id to a list of attribute labels calibrated for uncertainty
    # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
    uncertainty_map = {
        1: {
            1: 0,
            2: 0.5,
            3: 0.75,
            4: 1,
        },  # calibrate main label based on uncertainty label
        0: {1: 0, 2: 0.5, 3: 0.25, 4: 0},
    }
    with open(join(cwd, data_dir + "\\attributes\\image_attribute_labels.txt"), "r") as f:
        for line in f:
            file_idx, attribute_idx, attribute_label, attribute_certainty = (
                line.strip().split()[:4]
            )
            attribute_label = int(attribute_label)
            attribute_certainty = int(attribute_certainty)
            uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
            attribute_labels_all[int(file_idx)].append(attribute_label)
            attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
            attribute_certainties_all[int(file_idx)].append(attribute_certainty)

    is_train_test = dict()  # map from image id to 0 / 1 (1 = train)
    with open(join(cwd, data_dir + "\\train_test_split.txt"), "r") as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)
    print(
        "Number of train images from official train test split:",
        sum(list(is_train_test.values())),
    )
    train_val_data, test_data = [], []
    train_data, val_data = [], []
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()  # sort by class index
    for i, folder in enumerate(folder_list[:5]):
        folder_path = join(data_path, folder)
        classfile_list = [
            cf
            for cf in listdir(folder_path)
            if (isfile(join(folder_path, cf)) and cf[0] != ".")
        ]

        for cf in classfile_list:
            img_id = path_to_id_map[join(folder_path, cf)]
            img_path = join(folder_path, cf)
            metadata = {
                "id": img_id,
                "img_path": img_path,
                "img": dataset[i],
                "class_label": i,
                "attribute_label": torch.tensor(attribute_labels_all[img_id],dtype=torch.float32),
                "attribute_certainty": attribute_certainties_all[img_id],
                "uncertain_attribute_label": attribute_uncertain_labels_all[img_id],
            }
            if is_train_test[img_id]:
                train_val_data.append(metadata)
            else:
                test_data.append(metadata)

    random.shuffle(train_val_data)
    split = int(val_ratio * len(train_val_data))
    train_data = train_val_data[split:]
    val_data = train_val_data[:split]
    print("Size of train set:", len(train_data))
    return train_data, val_data, test_data

def data_load():
    #root_dir = f'{os.getcwd()}/CUB_200_2011/images/'
    root_dir = f'{os.getcwd()}\\CUB_200_2011\\images\\'

    transform = transforms.Compose([
        Resize((299, 299)),  # Resize images to a fixed size, for example, 224x224
        ToTensor()           # Convert images to tensors
    ])
    dataset = ImageFolder(root=root_dir,transform=transform)

    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining for testing

    # Split the dataset randomly into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    data_dir = f'{os.getcwd()}\\CUB_200_2011'
    train_dataset, val_dataset, test_dataset = extract_data(data_dir,dataset)
    

    return train_dataset, val_dataset, test_dataset

# Define your dataset and dataloader
# Assuming you have defined your dataset and dataloader somewhere in your code
train_dataset,val_dataset,test_dataset = data_load()

train_indices = list(range(len(train_dataset)))
val_indices = list(range(len(val_dataset)))
test_indices = list(range(len(test_dataset)))

random.shuffle(train_indices)
random.shuffle(val_indices)
random.shuffle(test_indices)

# Create SubsetRandomSamplers for all datasets
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)
model_c = ConceptModel()
model_c.load_state_dict(torch.load('indepc_model.pth'))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_c.model.fc.parameters(), lr=0.001)
# Define your model, optimizer, and loss function
#model.load_state_dict(torch.load('standard_model.pth'))
# Train the model
#train_c_model(model_c, train_loader, criterion, optimizer)

model_p = PredictionModel()
model_p.load_state_dict(torch.load('indepp_model.pth'))
optimizer = optim.Adam(model_p.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#train_p_model(model_p, train_loader, criterion, optimizer)
# Calculate accuracy on the validation set (assuming you have one)
# Assuming val_loader is defined somewhere in your code
val_accuracy_c,val_accuracy_p, = calculate_accuracy(model_c,model_p, test_loader)
print(f'Concept Accuracy: {val_accuracy_c:.4f}')
print(f'Prediction Accuracy: {val_accuracy_p:.4f}')

# Save the model
torch.save(model_c.state_dict(), 'indepc_model.pth')
torch.save(model_p.state_dict(), 'indepp_model2.pth')
