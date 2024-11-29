import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torchvision import  datasets

from functions import *
from additional_feature import *

import json
import os
import logging
import copy
import cv2
import time
from random import randint
import glob
import random
from PIL import Image
from ois import optimal_system
from torchvision.models import mobilenet_v3_small
from torch.utils.data import RandomSampler
from tqdm.contrib import tzip
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation,RandomVerticalFlip, RandomCrop, CenterCrop, Normalize, GaussianBlur, RandomRotation

# devices
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'CHECK devices {device}')


# Dataset
class All_samples(Dataset):

    def __init__(self, images, labels, mode=None, transform=None, rand_crop = None, local_scaler=None, default_size=20, bramich_img=None):


        self.labels = labels
        self.images = images
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.rand_crop = rand_crop
        self.local_scaler = local_scaler
        
        
    def __len__(self):

        input_size = np.shape(self.images)[0] 
        return input_size

    def __getitem__(self, idx):
        
        trk_imgs = self.images[idx].copy() # copy to avoid data modification
        trk_imgs = np.transpose(trk_imgs)
        label = self.labels[idx]
        
        
            
        
             
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
         
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)
            

   
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
                
            elif self.local_scaler == 'l2norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_L2Norm(local_scaled_imgs)
            
            elif self.local_scaler == 'bkg_std_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = sep_standard_scaler(local_scaled_imgs)
            
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
            
            
        
        if self.rand_crop is not None:

            imgs = RandomCrop(self.rand_crop)(imgs)


        else:
            imgs = CenterCrop(self.default_size)(imgs)

        
        
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            
            if 'bramich_diff' in self.mode:
                assert bramich_img!=None, f'Add bramich image'
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(imgs)
                imgs = np.concatenate((imgs,p2p_imgs), axis=0)
                
        
            
        return imgs, label
    
class Positive_samples(Dataset):

    def __init__(self, pos_images, mode=None, transform=None, rand_crop = None, local_scaler=None, default_size=20, bramich_img=None):


        #self.labels = np.ones(len(pos_images))
        self.pos_images = pos_images
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.rand_crop = rand_crop
        self.local_scaler = local_scaler
        
        
    def __len__(self):

        input_size = np.shape(self.pos_images)[0] 
        return input_size

    def __getitem__(self, idx):
        
        trk_imgs = self.pos_images[idx].copy() # copy to avoid data modification
        trk_imgs = np.transpose(trk_imgs)
        
        
            
        
             
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
         
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)
            

   
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
                
            elif self.local_scaler == 'l2norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_L2Norm(local_scaled_imgs)
            
            elif self.local_scaler == 'bkg_std_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = sep_standard_scaler(local_scaled_imgs)
            
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
            
            
        
        if self.rand_crop is not None:

            imgs = RandomCrop(self.rand_crop)(imgs)


        else:
            imgs = CenterCrop(self.default_size)(imgs)

        
        
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            
            if 'bramich_diff' in self.mode:
                assert bramich_img!=None, f'Add bramich image'
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(imgs)
                imgs = np.concatenate((imgs,p2p_imgs), axis=0)
                
        
            
        return imgs
    
class Positive_samples_withbkg(Dataset):

    def __init__(self, pos_images, mode=None, transform=None, rand_crop = None, local_scaler=None, default_size=20, add_similarity=None):


        #self.labels = np.ones(len(pos_images))
        self.pos_images = pos_images
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.rand_crop = rand_crop
        self.local_scaler = local_scaler
        self.add_sim = add_similarity
        
    def __len__(self):

        input_size = np.shape(self.pos_images)[0] 
        return input_size

    def __getitem__(self, idx):
        
        trk_imgs = self.pos_images[idx].copy() # copy to avoid data modification
        if self.add_sim is True:
            sim_imgs = get_similarity(trk_imgs)
            trk_imgs = np.concatenate([trk_imgs,sim_imgs], axis=0)
            
        trk_imgs = np.transpose(trk_imgs)
        
        
            
        
             
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
         
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)
            

   
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl','bkg_std_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
            
            elif self.local_scaler == 'bkg_std_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = sep_standard_scaler(local_scaled_imgs)
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
            
            
        
        if self.rand_crop is not None:

            imgs = RandomCrop(self.rand_crop)(imgs)


        else:
            imgs = CenterCrop(self.default_size)(imgs)

        
        
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            
            if 'bramich_diff' in self.mode:
                diff_imgs = get_bramich_diff(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(imgs)
                imgs = np.concatenate((imgs,p2p_imgs), axis=0)
                
        
            
        return imgs

    
# Dataset
class Negative_samples(Dataset):

    def __init__(self, neg_images, mode=None, transform=None, local_scaler=None, default_size=20,add_similarity=None,rand_crop=None):


        #self.labels = np.ones(len(pos_images))
        self.neg_images = neg_images
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.local_scaler = local_scaler
        self.rand_crop = rand_crop
        
        
        
    def __len__(self):

        input_size = np.shape(self.neg_images)[0] 
        return input_size

    def __getitem__(self, idx):
        
        
        trk_imgs = self.neg_images[idx].copy() # copy to avoid data modification
        trk_imgs = np.transpose(trk_imgs)
        
        
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
        
        
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)

        



                
                
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl','bkg_std_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
            
            elif self.local_scaler == 'l2norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_L2Norm(local_scaled_imgs)
                
            elif self.local_scaler == 'bkg_std_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = sep_standard_scaler(local_scaled_imgs)
            
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
        # Image cropping
        if self.rand_crop is not None:

            imgs = RandomCrop(self.rand_crop)(imgs)


        else:
            imgs = CenterCrop(self.default_size)(imgs)
        #imgs = CenterCrop(self.default_size)(imgs)
        #print(f'Size after cropping {trk_imgs.size()}')
        
        
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            
            if 'bramich_diff' in self.mode:
                diff_imgs = get_bramich_diff(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(imgs)
                imgs = np.concatenate((imgs,p2p_imgs), axis=0)
                
        
            
        return imgs
    
    
class Eval_set(Dataset):

    def __init__(self, pos_images, neg_images, mode=None, transform=None,  local_scaler=None, default_size=20,add_similarity=None):


        self.labels = np.concatenate((np.zeros(len(neg_images)), np.ones(len(pos_images))), axis=0)
        self.all_imgs = np.concatenate((neg_images, pos_images), axis=0)
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.local_scaler = local_scaler
        self.add_sim = add_similarity
        
        
    def __len__(self):

        input_size = np.shape(self.all_imgs)[0] 
        return input_size

    def __getitem__(self, idx):
        
        
        trk_imgs = self.all_imgs[idx].copy() # copy to avoid data modification
        if self.add_sim is True:
            sim_imgs = get_similarity(trk_imgs)
            trk_imgs = np.concatenate([trk_imgs,sim_imgs], axis=0)
        trk_imgs = np.transpose(trk_imgs)
        
            
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
         
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)
                    
                
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl','bkg_std_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
                
            elif self.local_scaler == 'l2norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_L2Norm(local_scaled_imgs)
                
            elif self.local_scaler == 'bkg_std_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = sep_standard_scaler(local_scaled_imgs)
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'    # No global scaler':
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
        # Center Cropping
        imgs = CenterCrop(self.default_size)(imgs)
        
        #print(f'CHECK IMAGE Shape {imgs.size()}')
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            
            if 'bramich_diff' in self.mode:
                diff_imgs = get_bramich_diff(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(imgs[:3]) # p2p done on raw only
                imgs = np.concatenate((imgs,p2p_imgs), axis=0)
                
        
            
        return imgs, self.labels[idx]


    

class Train_set(Dataset):

    def __init__(self, pos_images, neg_images, mode=None, transform=None,  local_scaler=None, default_size=20,add_similarity=None, rand_crop=None):


        self.labels = np.concatenate((np.zeros(len(neg_images)), np.ones(len(pos_images))), axis=0)
        self.all_imgs = np.concatenate((neg_images, pos_images), axis=0)
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.local_scaler = local_scaler
        self.add_sim = add_similarity
        self.rand_crop = rand_crop
        
        
    def __len__(self):

        input_size = np.shape(self.all_imgs)[0] 
        return input_size

    def __getitem__(self, idx):
        
        
        trk_imgs = self.all_imgs[idx].copy() # copy to avoid data modification
        if self.add_sim is True:
            sim_imgs = get_similarity(trk_imgs)
            trk_imgs = np.concatenate([trk_imgs,sim_imgs], axis=0)
        trk_imgs = np.transpose(trk_imgs)
        
            
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
         
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)
                    
                
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
            
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'    # No global scaler':
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
        # Image cropping
        if self.rand_crop is not None:

            imgs = RandomCrop(self.rand_crop)(imgs)


        else:
            imgs = CenterCrop(self.default_size)(imgs)
        
        #print(f'CHECK IMAGE Shape {imgs.size()}')
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            
            if 'bramich_diff' in self.mode:
                diff_imgs = get_bramich_diff(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(imgs[:3]) # p2p done on raw only
                imgs = np.concatenate((imgs,p2p_imgs), axis=0)
                
        
            
        return imgs, self.labels[idx]
        
# ======================================================================================================================= #
# Doig here
# Pair dataloader
class Pair_Dataset(torch.utils.data.IterableDataset):
    def __init__(self, pos_images, neg_images, shuffle_pairs=True, transform=None, default_size=20, rand_crop=None, mode=None):
        
        self.labels = np.concatenate((np.zeros(len(neg_images)), np.ones(len(pos_images))), axis=0)
        self.all_imgs = np.concatenate((neg_images, pos_images), axis=0)
        self.shuffle_pairs = shuffle_pairs
        self.transform = transform
        self.default_size = default_size
        self.rand_crop = rand_crop
        self.mode = mode
        self.create_pairs()

        
    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        #self.image_paths = glob.glob(os.path.join(self.path, "*/*.png"))
        self.image_classes = []
        self.class_indices = {}

        for img_idx, _ in enumerate(self.labels):
            image_class = self.labels[img_idx]
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            self.class_indices[image_class].append(img_idx)

        self.indices1 = np.arange(len(self.labels))

        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
            
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(42)

        select_pos_pair = np.random.rand(len(self.all_imgs)) < 0.5

        self.indices2 = []

        for i, pos in zip(self.indices1, select_pos_pair):
            class1 = self.image_classes[i]
            
            # pos means "same class"
            
            if pos:
                class2 = class1
            else:
                class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
                
            idx2 = np.random.choice(self.class_indices[class2])
            self.indices2.append(idx2)
        self.indices2 = np.array(self.indices2)
        

    def __iter__(self):
        self.create_pairs()

        for idx, idx2 in zip(self.indices1, self.indices2):

            image1 = np.transpose(self.all_imgs[idx])
            image2 = np.transpose(self.all_imgs[idx2])

            class1 = self.labels[idx]
            class2 = self.labels[idx2]


            if self.transform is not None:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
                
                
            local_scaled_imgs = torch.clone(image1)
            image1 = tensor_standard_scaler(local_scaled_imgs)
            
            local_scaled_imgs = torch.clone(image2)
            image2 = tensor_standard_scaler(local_scaled_imgs)
            
            
            if self.rand_crop is not None:

                #imgs = RandomCrop(self.rand_crop)(imgs)
                image1 = transforms.RandomCrop(self.rand_crop)(image1)
                image2 = transforms.RandomCrop(self.rand_crop)(image2)


            else:
                #imgs = CenterCrop(self.default_size)(imgs)
                image1 = transforms.CenterCrop(self.default_size)(image1)
                image2 = transforms.CenterCrop(self.default_size)(image2)
                
            if self.mode is None:
                pass


            else:
                
                assert self.mode in ['p2p'], 'The input mode is not found.'
                
                if 'p2p' in self.mode:
                    
                    # image 1
                    p2p_imgs_1 = get_p2p(image1[:3]) # p2p done on raw only
                    image1 = np.concatenate((image1,p2p_imgs_1), axis=0)
                    
                    # image 2
                    p2p_imgs_2 = get_p2p(image2[:3]) # p2p done on raw only
                    image2 = np.concatenate((image2,p2p_imgs_2), axis=0)

            
            
            
            

            yield (image1, image2), np.equal(class1,class2).astype(np.float32) , (class1, class2)
        
    def __len__(self):
        return len(self.all_imgs)
    
class Pair_Dataset_multicrops(torch.utils.data.IterableDataset):
    def __init__(self, pos_images, neg_images, shuffle_pairs=True, transform=None, default_size=20,rand_crop=None, num_crops = 1):
        
        self.labels = np.concatenate((np.zeros(len(neg_images)), np.ones(len(pos_images))), axis=0)
        self.all_imgs = np.concatenate((neg_images, pos_images), axis=0)
        self.shuffle_pairs = shuffle_pairs
        self.transform = transform
        self.default_size = default_size
        self.rand_crop = rand_crop
        self.num_crops = num_crops
        self.create_pairs()

        
    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        #self.image_paths = glob.glob(os.path.join(self.path, "*/*.png"))
        self.image_classes = []
        self.class_indices = {}

        for img_idx, _ in enumerate(self.labels):
            image_class = self.labels[img_idx]
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            self.class_indices[image_class].append(img_idx)

        self.indices1 = np.arange(len(self.labels))

        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
            
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(42)

        select_pos_pair = np.random.rand(len(self.all_imgs)) < 0.5

        self.indices2 = []

        for i, pos in zip(self.indices1, select_pos_pair):
            class1 = self.image_classes[i]
            
            # pos means "same class"
            
            if pos:
                class2 = class1
            else:
                class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
                
            idx2 = np.random.choice(self.class_indices[class2])
            self.indices2.append(idx2)
        self.indices2 = np.array(self.indices2)
        

    def __iter__(self):
        self.create_pairs()

        for idx, idx2 in zip(self.indices1, self.indices2):

            image1 = np.transpose(self.all_imgs[idx])
            image2 = np.transpose(self.all_imgs[idx2])

            class1 = self.labels[idx]
            class2 = self.labels[idx2]


            if self.transform is not None:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
                
                
            local_scaled_imgs = torch.clone(image1)
            image1 = tensor_standard_scaler(local_scaled_imgs)
            
            local_scaled_imgs = torch.clone(image2)
            image2 = tensor_standard_scaler(local_scaled_imgs)
            
            
            if self.rand_crop is not None:

                #imgs = RandomCrop(self.rand_crop)(imgs)
                image1 = transforms.RandomCrop(self.rand_crop)(image1)
                image2 = transforms.RandomCrop(self.rand_crop)(image2)


            else:
                #imgs = CenterCrop(self.default_size)(imgs)
                image1 = transforms.CenterCrop(self.default_size)(image1)
                image2 = transforms.CenterCrop(self.default_size)(image2)

            
            
            
            

            yield (image1, image2), np.equal(class1,class2).astype(np.float32) , (class1, class2)
        
    def __len__(self):
        return len(self.all_imgs)
    
    
class Pairing_SourceDataset_Train(torch.utils.data.IterableDataset):

    def __init__(self, images, label, mode=None, transform=None, shuffle_pairs=False, local_scaler=None, default_size=20, rand_crop=None):


        self.labels = label
        self.all_imgs = images
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.local_scaler = local_scaler
        self.rand_crop = rand_crop
        self.shuffle_pairs = shuffle_pairs
        
    def __len__(self):
        return np.shape(self.all_imgs)[0]
    
    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        self.indices1 = np.arange(len(self.labels))
        
        self.indices2 = np.arange(len(self.labels))
        
        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices2)
            
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(42)
            np.random.shuffle(self.indices2)


        
    def __iter__(self):
        self.create_pairs()

        for idx, idx2 in zip(self.indices1, self.indices2):
    
    
            # print(idx, idx2)
            image1 = np.transpose(self.all_imgs[idx])
            image2 = np.transpose(self.all_imgs[idx2])
            
            #image1 = transforms.ToTensor()(image1)
            #image2 = transforms.ToTensor()(image2)

            class1 = self.labels[idx]
            class2 = self.labels[idx2]


            if self.transform is not None:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
            
            # print(f'BEFORE IMG1: [{torch.min(image1), torch.max(image1)}]')
            # print(f'BEFORE IMG2: [{torch.min(image2), torch.max(image2)}]')
                
            local_scaled_imgs1 = torch.clone(image1)
            image1 = tensor_standard_scaler(local_scaled_imgs1)
            
            local_scaled_imgs2 = torch.clone(image2)
            image2 = tensor_standard_scaler(local_scaled_imgs2)
            
            
            if self.rand_crop is not None:

                #imgs = RandomCrop(self.rand_crop)(imgs)
                image1 = transforms.RandomCrop(self.rand_crop)(image1)
                image2 = transforms.RandomCrop(self.rand_crop)(image2)


            else:
                #imgs = CenterCrop(self.default_size)(imgs)
                image1 = transforms.CenterCrop(self.default_size)(image1)
                image2 = transforms.CenterCrop(self.default_size)(image2)

            # print(f'AFTER IMG1: [{torch.min(image1), torch.max(image1)}]')
            # print(f'AFTER IMG2: [{torch.min(image2), torch.max(image2)}]')
            
            
            

            yield (image1, image2), np.equal(class1,class2).astype(np.float32) , (class1, class2)
            
    
    
# ================ SupCon Dataset ================ #
# Dataset
class SupCon_Negative_samples(Dataset):

    def __init__(self, neg_images, mode=None, transform=None, local_scaler=None, default_size=20,add_similarity=None,num_crops=5):


        #self.labels = np.ones(len(pos_images))
        self.neg_images = neg_images
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.local_scaler = local_scaler
        self.add_sim = add_similarity
        self.rand_crop = default_size
        self.num_crops = num_crops
        
        
        
    def __len__(self):

        input_size = np.shape(self.neg_images)[0] 
        return input_size

    def __getitem__(self, idx):
        
        
        trk_imgs = self.neg_images[idx].copy() # copy to avoid data modification
        if self.add_sim is True:
            sim_imgs = get_similarity(trk_imgs)
            trk_imgs = np.concatenate([trk_imgs,sim_imgs], axis=0)
        trk_imgs = np.transpose(trk_imgs)
        
        
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
        
        
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)

        



                
                
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
                
            elif self.local_scaler == 'l2norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_L2Norm_SupCon(local_scaled_imgs)
            
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
        # Image cropping
        out_img = torch.empty((self.num_crops,imgs.size()[0],self.rand_crop,self.rand_crop), dtype=torch.float64)
        for i  in range(self.num_crops):
            out_img[i] = RandomCrop(self.rand_crop)(imgs)


       
        #imgs = CenterCrop(self.default_size)(imgs)
        #print(f'Size after cropping {trk_imgs.size()}')
        
        
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(out_img)
                out_img = np.concatenate((out_img,diff_imgs), axis=0)
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(out_img)
                out_img = np.concatenate((out_img,p2p_imgs), axis=0)
                
        
            
        return out_img
    
    
class SupCon_Positive_samples(Dataset):

    def __init__(self, pos_images, mode=None, transform=None, local_scaler=None, default_size=20, add_similarity=None,num_crops=5):


        #self.labels = np.ones(len(pos_images))
        self.pos_images = pos_images
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.rand_crop = default_size
        self.local_scaler = local_scaler
        self.add_sim = add_similarity
        self.num_crops = num_crops
        
    def __len__(self):

        input_size = np.shape(self.pos_images)[0] 
        return input_size

    def __getitem__(self, idx):
        
        trk_imgs = self.pos_images[idx].copy() # copy to avoid data modification
        if self.add_sim is True:
            sim_imgs = get_similarity(trk_imgs)
            trk_imgs = np.concatenate([trk_imgs,sim_imgs], axis=0)
            
        trk_imgs = np.transpose(trk_imgs)
        
        
            
        
             
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
         
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)
            

   
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
            
            elif self.local_scaler == 'l2norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_L2Norm_SupCon(local_scaled_imgs)
            
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
            
            
        
        # Image cropping
        out_img = torch.empty((self.num_crops,imgs.size()[0],self.rand_crop,self.rand_crop), dtype=torch.float64)
        for i  in range(self.num_crops):
            out_img[i] = RandomCrop(self.rand_crop)(imgs)

        
        
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(out_img)
                out_img = np.concatenate((out_img,diff_imgs), axis=0)
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(out_img)
                out_img = np.concatenate((out_img,p2p_imgs), axis=0)
                
        
            
        return out_img
    
class SupCon_Eval_set(Dataset):

    def __init__(self, pos_images, neg_images, mode=None, transform=None,  local_scaler=None, default_size=20,add_similarity=None, num_crops = 5):


        self.labels = np.concatenate((np.zeros(len(neg_images)), np.ones(len(pos_images))), axis=0)
        self.all_imgs = np.concatenate((neg_images, pos_images), axis=0)
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.local_scaler = local_scaler
        self.add_sim = add_similarity
        self.num_crops = num_crops
        self.rand_crop = default_size
        
        
        
        
    def __len__(self):

        input_size = np.shape(self.all_imgs)[0] 
        return input_size

    def __getitem__(self, idx):
        
        
        trk_imgs = self.all_imgs[idx].copy() # copy to avoid data modification
        if self.add_sim is True:
            sim_imgs = get_similarity(trk_imgs)
            trk_imgs = np.concatenate([trk_imgs,sim_imgs], axis=0)
        trk_imgs = np.transpose(trk_imgs)
        
            
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
         
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)
                    
                
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
                
            elif self.local_scaler == 'l2norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_L2Norm_SupCon(local_scaled_imgs)
            
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'    # No global scaler':
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
        # Image cropping
        out_img = torch.empty((1,imgs.size()[0],self.rand_crop,self.rand_crop), dtype=torch.float64)
        
        out_img[0] = CenterCrop(self.default_size)(imgs)

        
        
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(out_img)
                out_img = np.concatenate((out_img,diff_imgs), axis=0)
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(out_img)
                out_img = np.concatenate((out_img,p2p_imgs), axis=0)
                
        
            
        return out_img, self.labels[idx]
