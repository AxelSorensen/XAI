import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class XtoCDataset(Dataset):
    def __init__(self, metadata_list, transform=None):
        self.metadata_list = metadata_list
        self.transform = transform
        self.images = []  # List to store preloaded images
        self.class_labels = []  # List to store labels
        self.attribute_labels = []  # List to store attributes
        self.ids = []  # List to store indices
        
        for bird in metadata_list: # bird = item
            img_path = bird['img_path']
            image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB
            if self.transform:
                image = self.transform(image)

            self.images.append(image)
            self.class_labels.append(bird['class_label'])
            self.attribute_labels.append(bird['attribute_label'])
            self.ids.append(bird['id'])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Return preloaded image and label
        image = self.images[idx]
        class_label = self.class_labels[idx]
        attr_label = self.attribute_labels[idx]        
        return image, class_label, attr_label


class ImageDatasetWithMetadata(Dataset):
    def __init__(self, metadata_list, transform=None):
        """
        Args:
            metadata_list (list of dicts): List containing metadata including image paths.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.metadata_list = metadata_list
        self.transform = transform

    def __len__(self):
        return len(self.metadata_list)
    
    def __getitem__(self, idx):
        # Get metadata for the specified index
        metadata = self.metadata_list[idx]
        
        # Load image
        img_path = metadata['img_path']
        image = Image.open(img_path)
        
        # Apply transformations to the image (if any)
        if self.transform:
            image = self.transform(image)
        
        # Get class label and attribute label
        class_label = torch.tensor(metadata['class_label'], dtype=torch.long)
        attribute_label = torch.tensor(metadata['attribute_label'], dtype=torch.float)
        
        # Return image and its metadata
        return image, class_label, attribute_label