from torchvision import transforms as T
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np


class AlbumentationTransforms:
    """
    Helper class to create test and train transforms using Albumentations
    """
    def __init__(self, transforms_list=[]):
        transforms_list.append(AP.ToTensor())

        self.transforms = A.Compose(transforms_list)
        print("Composed all transforms")


    def __call__(self, img):
        img = np.array(img)
        trans = self.transforms(image=img)['image']
        #print("Applied all transforms")
        
        return trans
