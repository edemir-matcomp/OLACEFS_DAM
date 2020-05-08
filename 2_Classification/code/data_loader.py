import time
import os, tqdm, random
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from torch.utils import data
from matplotlib.pyplot import imsave
import torchvision.transforms as T
from torchvision import datasets, models, transforms
from skimage.io import imread
from skimage.transform import resize
import skimage.transform as scikit_transform


filter_classes = ['']

def get_dataset_base(root, channels, path_classes):

    info_dict = {}
    
    to_remove = []
     
    
    # Get images path and labels
    full_list_images = recursive_glob(root)

    list_images = full_list_images
        
    # BARRAGEM E NAO_BARRAGEM
    if path_classes == 0:
    
        # Create dict of classes
        classes = sorted(list(set([i.split('/')[-2] for i in list_images])))   
        labels_dict = {}
        for id, c in enumerate(classes):
            labels_dict[c] = id
            
        list_labels = [labels_dict[i.split('/')[-2]] for i in list_images]
    
    
    # MINÉRIO PRINCIPAL
    elif path_classes == 1:
    
        # load table data
        table = np.loadtxt('code/config/dam_2019_table.tsv', delimiter='\t' , dtype='str', usecols=(0,9), skiprows=1, encoding='utf8')

        # get all labeled samples
        list_labels = [i for i in table[:,1]]

        # filter classes only with more than 10 samples_weights
        counts = np.unique(list_labels, return_counts=True)
        filtered_classes = list(counts[0][counts[1] >= 10])
        filtered_classes.append('Others')

        labels_dict = {}
        classes = sorted(filtered_classes)
        for id, c in enumerate(filtered_classes):
            labels_dict[c] = id

        # get all images
        list_images = []
        list_labels = []
        for image, label in (table[:,:]):

            list_images.append(root+'/barragem/'+str(image).zfill(3)+'_2019.tif')
            
            #list_images.append('/home/users/edemir/barragem_detection/data/DATASET_BARRAGEM_LASTONE2/new_sentinel/processed/2019/barragem/'+str(image).zfill(3)+'_2019.tif.npy')
            
            # check if in filtered_classes
            if label in filtered_classes:
                list_labels.append(labels_dict[label])
            else:
                list_labels.append(labels_dict['Others'])

    # MÉTODO CONSTRUTIVO
    elif path_classes == 2:
    
        table = np.loadtxt('code/config/dam_2019_table.tsv', delimiter='\t' , dtype='str', usecols=(0,12), skiprows=1, encoding='utf8')

        # get all labeled samples
        list_labels = [i for i in table[:,1]]

        # filter classes only with more than 10 samples_weights
        counts = np.unique(list_labels, return_counts=True)
        print(counts)
        filtered_classes = list(counts[0])
        classes = sorted(filtered_classes)

        labels_dict = {}
        for id, c in enumerate(filtered_classes):
            labels_dict[c] = id

        # get all images
        list_images = []
        list_labels = []
        for image, label in (table[:,:]):

            list_images.append(root+'/barragem/'+str(image).zfill(3)+'_2019.tif')
            list_labels.append(labels_dict[label])
            
    # CATEGORIA DE RISCO
    elif path_classes == 3:
    
        table = np.loadtxt('code/config/dam_2019_table.tsv', delimiter='\t' , dtype='str', usecols=(0,13), skiprows=1, encoding='utf8')

        # get all labeled samples
        list_labels = [i for i in table[:,1]]

        # filter classes only with more than 10 samples_weights
        counts = np.unique(list_labels, return_counts=True)
        print(counts)
        filtered_classes = list(counts[0][[2,3]])
        classes = sorted(filtered_classes)
        #filtered_counts = counts[1][counts[1] >= 10]

        labels_dict = {}
        for id, c in enumerate(filtered_classes):
            labels_dict[c] = id

        # get all images
        list_images = []
        list_labels = []
        for image, label in (table[:,:]):

            # check if in filtered_classes
            if label in filtered_classes:
                list_images.append(root+'/barragem/'+str(image).zfill(3)+'_2019.tif')
                list_labels.append(labels_dict[label])
    
    
    # DANO POTENCIAL ASSOCIADO
    elif path_classes == 4:
    
        table = np.loadtxt('code/config/dam_2019_table.tsv', delimiter='\t' , dtype='str', usecols=(0,14), skiprows=1, encoding='utf8')

        # get all labeled samples
        list_labels = [i for i in table[:,1]]

        # filter classes only with more than 10 samples_weights
        counts = np.unique(list_labels, return_counts=True)
        filtered_classes = list(counts[0][[1,2,3]])
        #filtered_classes.append('Others')
        #filtered_counts = counts[1][counts[1] >= 10]
        classes = sorted(filtered_classes)

        labels_dict = {}
        for id, c in enumerate(filtered_classes):
            labels_dict[c] = id

        # get all images
        list_images = []
        list_labels = []
        for image, label in (table[:,:]):

            # check if in filtered_classes
            if label in filtered_classes:
            
                list_images.append(root+'/barragem/'+str(image).zfill(3)+'_2019.tif')
                list_labels.append(labels_dict[label])
    
    
    info_dict['list_images'] = list_images
    info_dict['list_labels'] = list_labels
    info_dict['labels_dict'] = labels_dict
    info_dict['classes'] = classes
    info_dict['len'] = len(list_labels)
    
    print('CHANNELS: {}'.format(channels))
    if channels == 'rgb':
        info_dict['channels'] = [3,2,1]
    elif channels == 'rgb_wa':
        info_dict['channels'] = list(range(3))
    elif channels == 'spectral':
        info_dict['channels'] = list(range(13))
    else:
    
        print(list_images[0])
        # Load image to retrieve channels
        try:
            img = imread(list_images[0])
        except:
            img = np.load(list_images[0])
            
        _, _, c = img.shape
    
        info_dict['channels'] = list(range(c))   #c
    
    print(info_dict['channels'])
    
    #print(info_dict)
        
    return info_dict

def recursive_glob(rootdir=".", suffix=".tif"):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot,filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix) or filename.endswith('.png') or filename.endswith('.npy')
    ]

class Remote_Sensing_Loader(data.Dataset):
    def __init__(self, dataset_info, indices, transform=None):
            
        # Get images path and labels
        self.list_images = [dataset_info['list_images'][i] for i in indices]
        self.list_labels = [dataset_info['list_labels'][i] for i in indices]
        self.len = len(self.list_images)
        
        self.mean = None
        self.std = None
        
        # Create dict of classes
        self.classes = dataset_info['classes']
        self.num_classes = len(self.classes)        
        self.labels_dict = dataset_info['labels_dict']

        self.weight_class = 1. / np.unique(np.array(self.list_labels), return_counts=True)[1]
        self.samples_weights = self.weight_class[self.list_labels]
        
        self.channels = dataset_info['channels']
        
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        
        # Get new image path
        img_path = self.list_images[index]
        lbl = self.list_labels[index]
        
        # Load image and label
        try:
            img = imread(img_path)[:, :, self.channels]
            if img.shape[0] != 384 or img.shape[1] != 384:
                img = scikit_transform.resize(img, (384,384)).astype(img.dtype)
            
        except:
            img = np.load(img_path)[:, :, self.channels]

        # Random augmentation
        aug_choice = np.random.randint(5)

        #if random.random() < 0.5:
        
        
        if aug_choice == 0:
            # Rotate +90
            img = np.rot90(img).copy()
        elif aug_choice == 1:
            #Flip an array horizontally.
            img = np.fliplr(img).copy()
        elif aug_choice == 2:
            #Flip an array horizontally.
            img = np.flipud(img).copy()
        elif aug_choice == 3:
            # Rotate -90
            img = np.rot90(img,k=3).copy()
            
        
        # Apply transform 
        if self.transform:
            img = self.transform(img).float()
            #img = self.transform(T.functional.to_pil_image(img)).float()
                
        return img, lbl, img_path

'''
def add_embedded():

    out = torch.cat((outputs.data, torch.ones(len(outputs), 1).cuda()), 1)
                
    # Inverse Normalization
    rev_inputs = inputs.new(*inputs.size())
    for c in range(out.size(1)):
        rev_inputs[:, c, :, :] = inputs[:, c, :, :] * std[c] + mean[c]
        #rev_inputs[:, c, :, :] *= 255. / rev_inputs[:, c, :, :].max() 
   
    
    writer.add_embedding(
        out,
        metadata=labels.tolist(),
        label_img=rev_inputs,
        global_step=epoch)
'''

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = None
    snd_moment = None

    print("Computing mean of data...")
    for data_batch, _, _ in tqdm.tqdm(loader):
        b, c, h, w = data_batch.shape
        
        data_batch = data_batch.double()
        
        
        for it in range(data_batch.shape[0]):
        
            data = data_batch[it].view(1,c,h,w)
            #print(data.shape)
        
            if fst_moment is None:
                fst_moment = torch.empty(c, dtype=torch.double)
                snd_moment = torch.empty(c, dtype=torch.double)

            nb_pixels = b * h * w

            sum_ = torch.sum(data, dim=[0, 2, 3])
            sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def get_transforms(loader, phase, input_size, mean=None, std=None):

    # Load dataloader for train and test
    loader.dataset.transform = transforms.Compose([#transforms.Resize(input_size), 
                                                    #transforms.CenterCrop(input_size), 
                                                    transforms.ToTensor()])
    
    if phase == 'train':
        
        if mean is None:
            mean, std = online_mean_and_sd(loader)
            print(mean,std)
        
        transform = transforms.Compose([#transforms.RandomResizedCrop(input_size),
                                                        #transforms.Resize(input_size), 
                                                        #transforms.RandomHorizontalFlip(), 
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean, std)
                                                        ])
    else:
        transform = transforms.Compose([#transforms.Resize(input_size), 
                                                        #transforms.CenterCrop(input_size), 
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean, std)
                                                        ])
    return mean, std, transform

