from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image
import torch
from sklearn import preprocessing
import torch

class TrafficDataset(Dataset):
    """ 
        Dataset generator for Segmentation(Mask Prediction)
        and Depth Prediction 
    """
    
    def __init__(self, img_dir, labels, type_data, transform=None):
        """
        Args:
            img_dir = images directory. (Actual dataset)
            labels = labels for all images
            type_data = train or test
            transform = Augmentations list applied on images
        """
        
        self.img_dir = img_dir
        self.img_ids = []
        self.labels = []
        self.foldnm = labels
        self.transform = transform
        
        #storing all the folder names inside trafficnet dataset
        self.folders = [file for file in os.listdir(self.img_dir) if not file.startswith('.')]

        #converting labels into tensor
        lb = preprocessing.LabelEncoder()
        targets = lb.fit_transform(labels)
        targets = torch.as_tensor(targets)

        #iterating over each folder:
        for folder in self.folders:
            path = os.path.join(self.img_dir, folder)
            if(folder == labels[0]):
                labl = targets[0]
            elif(folder == labels[1]):
                labl = targets[1]
            elif(folder == labels[2]):
                labl = targets[2]
            elif(folder == labels[3]):
                labl = targets[3]
            else:
                labl = "no label"

            #getting img ids and labels in 2 lists
            for img in os.listdir(path):
                self.img_ids.append(img)
                self.labels.append(labl) 

        #shuffle imaegs and labels
        zipped = list(zip(self.img_ids, self.labels))
        random.shuffle(zipped)

        self.imgs, self.labls = zip(*zipped)
        print("Shuffled all data")


        if type_data == "train":
            print("Train Data Length:{}".format(len(self.imgs)))
            print("Train Labels length:{}".format(len(self.labls)))

        if type_data == "test":
            print("Test Data Shape:{}".format(len(self.imgs)))
            print("Test Labels length:{}".format(len(self.labls)))
        
    
    def __len__(self):
        return len(self.img_ids)


    def shuffle(self):
        zipped = list(zip(self.img_ids, self.labels))
        random.shuffle(zipped)

        imgs, labls = zip(*zipped)

        return imgs, labls
        
    
    def __getitem__(self, index):
        idx = self.imgs[index]
        label = self.labls[index]
        fold = int(label.numpy())

        if(fold == 0):
          folder = self.foldnm[0]
        elif(fold == 1):
          folder = self.foldnm[1]
        elif(fold == 2):
          folder = self.foldnm[2]
        else:
          folder = self.foldnm[3]
        
        
        data = Image.open(self.img_dir + folder + "/" + idx)
        
        """
        #image_tensor = torch.FloatTensor(image)
        #img =torch.from_numpy(image).permute(2,1,0)
        #img_tensor = torch.FloatTensor(img)
        #print("sape of image is:{}".format(img.shape))
        #y_label = self.labels[index]
        """
        
        
        if self.transform:
            image = self.transform(data)

        
        return image, label

class IFODataset(Dataset):
    """ 
        Dataset generator for MobileNetV2 implementation on Identified
        flying objects dataset
    """
    
    def __init__(self, data, labels, categories, type_data, transform=None):
        """
        Args:
            img_dir = images directory. (Actual dataset)
            labels = labels for all images
            type_data = train or test
            transform = Augmentations list applied on images
        """
        
        self.img_ids = data
        self.labels = labels
        self.category = categories
        self.transform = transform
        
        #storing all the folder names inside  IFOdataset
        #self.folders = [file.split('/') for file in os.listdir(self.img_dir) if not file.startswith('.')]

        #converting labels into tensor
        lb = preprocessing.LabelEncoder()
        targets = lb.fit_transform(categories)
        targets = torch.as_tensor(targets)

        #iterating over the labels list converting each into tensor:
        for idx, label in enumerate(self.labels):
            if(label == categories[0]):
                labl = targets[0]
            elif(label == categories[1]):
                labl = targets[1]
            elif(label == categories[2]):
                labl = targets[2]
            elif(label == categories[3]):
                labl = targets[3]
            else:
                labl = "no label"

            #substituting corresponding values in labels list with tensor label
            self.labels[idx] = labl


        if type_data == "train":
            print("Train Data length:{}".format(len(self.img_ids)))
            print("Train Labels length:{}".format(len(self.labels)))

        if type_data == "test":
            print("Test Data length:{}".format(len(self.img_ids)))
            print("Test Labels length:{}".format(len(self.labels)))
        
    
    def __len__(self):
        return len(self.img_ids)

    
    def __getitem__(self, index):
        idx = self.img_ids[index]
        label = self.labels[index]

        
        data = Image.open(idx)
        
        """
        #image_tensor = torch.FloatTensor(image)
        #img =torch.from_numpy(image).permute(2,1,0)
        #img_tensor = torch.FloatTensor(img)
        #print("shape of image is:{}".format(img.shape))
        #y_label = self.labels[index]
        """
        
        
        if self.transform:
            image = self.transform(data)

        
        return image, label
 