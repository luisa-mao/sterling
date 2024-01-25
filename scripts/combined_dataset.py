import sys
import os

# Add the top-level project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import numpy as np
import pickle
import glob
from torch.utils.data import DataLoader, Dataset, ConcatDataset
# import torch.nn.functional as F
from scipy.signal import periodogram
from scripts.utils import process_feet_data, get_transforms
from termcolor import cprint
import yaml
import cv2
import pytorch_lightning as pl
import os
from tqdm import tqdm
import time
from PIL import Image
from PIL import Image
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

FEET_TOPIC_RATE = 24.0
LEG_TOPIC_RATE = 24.0
IMU_TOPIC_RATE = 200.0

class TerrainDataset(Dataset):
    def __init__(self, pickle_files_roots, train=False):
        self.pickle_files_paths = []
        self.len = 0
        for root in pickle_files_roots:
            tmp = glob.glob(os.path.join(root, '*.pkl'))
            self.pickle_files_paths.append(tmp)
            self.len += len(tmp)
        
        self.labels = [root.split('/')[-2] for root in pickle_files_roots]
        if train:
            self.transforms = get_transforms()
        else:
            self.transforms = None

        self.load()


    def load(self):
        # load in all the data from pickle file paths in memory
        self.dataset = []
        for paths in self.pickle_files_paths:
            bag = []
            self.dataset.append(bag)
            for path in paths:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                bag.append(data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        pass

class TripletDataset(TerrainDataset):
    def __init__(self, pickle_files_roots, train=False):
        super().__init__(pickle_files_roots, train=train)
    
    def __getitem__(self, idx):
        # print("GET ITEM TRIPLET")
        idx = idx % self.len
        label_idx = 0
        i = idx
        while i >= len(self.pickle_files_paths[label_idx]):
            i -= len(self.pickle_files_paths[label_idx])
            label_idx += 1
        # with open(self.pickle_files_paths[label_idx][i], 'rb') as f:
        #     data = pickle.load(f)
        data = self.dataset[label_idx][i]
        patches = data['patches']
    
        # sample a number in the patches
        patch_1_idx = np.random.randint(0, len(patches))
        # get a positive patch
        pos_sample_idx = np.random.randint(0, len(self.pickle_files_paths[label_idx]))
        with open(self.pickle_files_paths[label_idx][pos_sample_idx], 'rb') as f:
            data = pickle.load(f)
        pos_patches = data['patches']
        pos_patch_idx = np.random.randint(0, len(pos_patches))

        # get a negative patch
        neg_label = np.random.randint(0, len(self.labels))
        while self.labels[neg_label] == self.labels[label_idx]:
            neg_label = np.random.randint(0, len(self.labels))
        neg_sample_idx = np.random.randint(0, len(self.pickle_files_paths[neg_label]))
        with open(self.pickle_files_paths[neg_label][neg_sample_idx], 'rb') as f:
            data = pickle.load(f)
        neg_patches = data['patches']
        neg_patch_idx = np.random.randint(0, len(neg_patches))
        
        patch1, patch2 = patches[patch_1_idx], pos_patches[pos_patch_idx]
        neg_patch = neg_patches[neg_patch_idx]
        
        # convert BGR to RGB
        patch1, patch2 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB), cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
        neg_patch = cv2.cvtColor(neg_patch, cv2.COLOR_BGR2RGB)
        
        # apply the transforms
        if self.transforms is not None:
            patch1 = self.transforms(image=patch1)['image']
            patch2 = self.transforms(image=patch2)['image']
            neg_patch = self.transforms(image=neg_patch)['image']
        
        # normalize the image patches
        patch1 = np.asarray(patch1, dtype=np.float32) / 255.0
        patch2 = np.asarray(patch2, dtype=np.float32) / 255.0
        neg_patch = np.asarray(neg_patch, dtype=np.float32) / 255.0
        
        # transpose
        patch1, patch2 = np.transpose(patch1, (2, 0, 1)), np.transpose(patch2, (2, 0, 1))
        neg_patch = np.transpose(neg_patch, (2, 0, 1))
        
        return np.asarray(patch1), np.asarray(patch2), np.asarray(neg_patch), self.labels[label_idx], idx

class VicregDataset(TerrainDataset):
    def __init__(self, pickle_files_roots, incl_orientation=False, 
                    data_stats=None, train=False):
        super().__init__(pickle_files_roots, train=train)
        self.data_stats = data_stats
        self.incl_orientation = incl_orientation
        if self.data_stats is not None:
            self.min, self.max = data_stats['min'], data_stats['max']
            self.mean, self.std = data_stats['mean'], data_stats['std']
    def __getitem__(self, idx):
        # print("GET ITEM VICREG")

        idx = idx % self.len
        label_idx = 0
        i = idx
        while i >= len(self.pickle_files_paths[label_idx]):
            i -= len(self.pickle_files_paths[label_idx])
            label_idx += 1
        # with open(self.pickle_files_paths[label_idx][i], 'rb') as f:
        #     data = pickle.load(f)
        data = self.dataset[label_idx][i]
        patches = data['patches']
        with open(self.pickle_files_paths[label_idx][i], 'rb') as f:
            data = pickle.load(f)
        imu, feet, leg = data['imu'], data['feet'], data['leg']
        patches = data['patches']
    
        # process the feet data to remove the mu and std values for non-contacting feet
        feet = process_feet_data(feet)
        
        if not self.incl_orientation: imu = imu[:, :-4]

        imu = periodogram(imu, fs=IMU_TOPIC_RATE, axis=0)[1]
        leg = periodogram(leg, fs=LEG_TOPIC_RATE, axis=0)[1]
        feet = periodogram(feet, fs=FEET_TOPIC_RATE, axis=0)[1]
        
        # normalize the imu data
        # if self.mean is not None and self.std is not None:
        if self.data_stats is not None:
            # #minmax normalization
            imu = (imu - self.min['imu']) / (self.max['imu'] - self.min['imu'] + 1e-7)
            imu = imu.flatten()
            imu = imu.reshape(1, -1)
            
            leg = (leg - self.min['leg']) / (self.max['leg'] - self.min['leg'] + 1e-7)
            leg = leg.flatten()
            leg = leg.reshape(1, -1)
            
            feet = (feet - self.min['feet']) / (self.max['feet'] - self.min['feet'] + 1e-7)
            feet = feet.flatten()
            feet = feet.reshape(1, -1)
                    
        # sample a number between 0 and (num_patches-1)/2
        patch_1_idx = np.random.randint(0, len(patches)//2)
        # sample a number between (num_patches-1)/2 and num_patches-1
        patch_2_idx = np.random.randint(len(patches)//2, len(patches))
        patch1, patch2 = patches[patch_1_idx], patches[patch_2_idx]
        
        # convert BGR to RGB
        patch1, patch2 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB), cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
        
        # apply the transforms
        if self.transforms is not None:
            patch1 = self.transforms(image=patch1)['image']
            patch2 = self.transforms(image=patch2)['image']
        
        # normalize the image patches
        patch1 = np.asarray(patch1, dtype=np.float32) / 255.0
        patch2 = np.asarray(patch2, dtype=np.float32) / 255.0
        
        # transpose
        patch1, patch2 = np.transpose(patch1, (2, 0, 1)), np.transpose(patch2, (2, 0, 1))
        
        return np.asarray(patch1), np.asarray(patch2), imu, leg, feet, self.labels[label_idx], idx


############################################################################
#   diff distributions
############################################################################
class SplitTerrainDataModule(Dataset):
    def __init__(self, pickle_files_roots, incl_orientation=False, 
                    data_stats=None, train=False):
        super().__init__(pickle_files_roots, train=train)
        # read the yaml file
        cprint('Reading the yaml file at : {}'.format(data_config_path), 'green')
        self.data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
        self.data_config_path = '/'.join(data_config_path.split('/')[:-1])

        self.include_orientation_imu = include_orientation_imu

        self.batch_size, self.num_workers = batch_size, num_workers
        
        self.mean, self.std = {}, {}
        self.min, self.max = {}, {}

        self.triplet_ds = None
        self.vicreg_ds = None
        
        # load the train and val datasets
        self.load()

        # print the lengths and true lengths of the datasets
        print("Triplet Train dataset length : {}".format(self.triplet_train_dataset.len))
        print("Triplet Val dataset length : {}".format(self.triplet_val_dataset.len))
        print("Vicreg Train dataset length : {}".format(self.vicreg_train_dataset.len))
        print("Vicreg Val dataset length : {}".format(self.vicreg_val_dataset.len))

        
    def load(self):
        # load the train data
        if train:
            file_roots = [pickle_files_root for pickle_files_root in self.data_config['train']]
        else:
            file_roots = [pickle_files_root for pickle_files_root in self.data_config['val']]
        triplet_loss_terrains = set(terrain for terrain in self.data_config['triplet_loss'])
        vicreg_loss_terrains = [terrain for terrain in self.data_config['vicreg_loss']]

        triplet_loss_data = []
        vicreg_loss_data = []

        for string in file_roots:
            if string.split('/')[-2] in triplet_loss_terrains:
                triplet_loss_data.append(string)
            if string.split('/')[-2] in vicreg_loss_terrains:
                vicreg_loss_data.append(string)
        # print( 'data config path: ', self.data_config_path)

        # check if the data_statistics.pkl file exists
        if os.path.exists(self.data_config_path + '/data_statistics.pkl'):
            cprint('Loading the mean and std from the data_statistics.pkl file', 'green')
            data_statistics = pickle.load(open(self.data_config_path + '/data_statistics.pkl', 'rb'))
            self.mean, self.std = data_statistics['mean'], data_statistics['std']
            self.min, self.max = data_statistics['min'], data_statistics['max']
            
        else:
            # find the mean and std of the train dataset
            cprint('data_statistics.pkl file not found!', 'yellow')
            cprint('Finding the mean and std of the train dataset', 'green')
            self.tmp_dataset = ConcatDataset([VicregDataset(vicreg_loss_train_data+vicreg_loss_val_data, incl_orientation=self.include_orientation_imu) for pickle_files_root in self.data_config['train']])
            self.tmp_dataloader = DataLoader(self.tmp_dataset, batch_size=128, num_workers=2, shuffle=False)
            cprint('the length of the tmp_dataloader is : {}'.format(len(self.tmp_dataloader)), 'green')
            # find the mean and std of the train dataset
            imu_data, leg_data, feet_data = [], [], []
            for _, _, imu, leg, feet, _, _ in tqdm(self.tmp_dataloader):
                imu_data.append(imu.cpu().numpy())
                leg_data.append(leg.cpu().numpy())
                feet_data.append(feet.cpu().numpy())
            imu_data = np.concatenate(imu_data, axis=0)
            leg_data = np.concatenate(leg_data, axis=0)
            feet_data = np.concatenate(feet_data, axis=0)
            print('imu_data.shape : ', imu_data.shape)
            print('leg_data.shape : ', leg_data.shape)
            print('feet_data.shape : ', feet_data.shape)
            # exit() # why?
            
            imu_data = imu_data.reshape(-1, imu_data.shape[-1])
            leg_data = leg_data.reshape(-1, leg_data.shape[-1])
            feet_data = feet_data.reshape(-1, feet_data.shape[-1])
            
            self.mean['imu'], self.std['imu'] = np.mean(imu_data, axis=0), np.std(imu_data, axis=0)
            self.min['imu'], self.max['imu'] = np.min(imu_data, axis=0), np.max(imu_data, axis=0)
            
            self.mean['leg'], self.std['leg'] = np.mean(leg_data, axis=0), np.std(leg_data, axis=0)
            self.min['leg'], self.max['leg'] = np.min(leg_data, axis=0), np.max(leg_data, axis=0)
            
            self.mean['feet'], self.std['feet'] = np.mean(feet_data, axis=0), np.std(feet_data, axis=0)
            self.min['feet'], self.max['feet'] = np.min(feet_data, axis=0), np.max(feet_data, axis=0)
            
            cprint('Mean : {}'.format(self.mean), 'green')
            cprint('Std : {}'.format(self.std), 'green')
            cprint('Min : {}'.format(self.min), 'green')
            cprint('Max : {}'.format(self.max), 'green')
            
            # save the mean and std
            cprint('Saving the mean, std, min, max to the data_statistics.pkl file', 'green')
            data_statistics = {'mean': self.mean, 'std': self.std, 'min': self.min, 'max': self.max}
            
            pickle.dump(data_statistics, open(self.data_config_path + '/data_statistics.pkl', 'wb'))
            
        # load the train data

        self.triplet_ds = TripletDataset(triplet_loss_data, train)
        self.vicreg_train_dataset = VicregDataset(vicreg_train_data, incl_orientation=self.include_orientation_imu, data_stats=data_statistics, train=train)
    def __getitem__(self, idx):
        return self.triplet_ds[idx], self.vicreg_ds[idx]
    def __len__(self):
        return max(len(self.triplet_ds), len(self.vicreg_ds))



# read terrains distribution from config files
class SplitTerrainDataModule(pl.LightningDataModule):
    def __init__(self, data_config_path, batch_size=64, num_workers=2, include_orientation_imu=False):
        super().__init__()
        
        # read the yaml file
        cprint('Reading the yaml file at : {}'.format(data_config_path), 'green')
        self.data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
        self.data_config_path = '/'.join(data_config_path.split('/')[:-1])

        self.include_orientation_imu = include_orientation_imu

        self.batch_size, self.num_workers = batch_size, num_workers
        
        self.mean, self.std = {}, {}
        self.min, self.max = {}, {}

        self.triplet = None
        self.vicreg = None
        
        # load the train and val datasets
        self.load()
        # cprint('Triplet Train dataset size : {}'.format(len(self.triplet_train_dataset)), 'green')
        # cprint('Triplet Val dataset size : {}'.format(len(self.triplet_val_dataset)), 'green')
        # cprint('Vicreg Train dataset size : {}'.format(len(self.vicreg_train_dataset)), 'green')
        # cprint('Vicreg Val dataset size : {}'.format(len(self.vicreg_val_dataset)), 'green')

        # set the lengths of the datasets to the max of the two
        # train_len = max(len(self.triplet_train_dataset), len(self.vicreg_train_dataset))
        # self.triplet_train_dataset.len = train_len
        # self.vicreg_train_dataset.len = train_len
        # val_len = max(len(self.triplet_val_dataset), len(self.vicreg_val_dataset))
        # self.triplet_val_dataset.len = val_len
        # self.vicreg_val_dataset.len = val_len

        # print the lengths and true lengths of the datasets
        print("Triplet Train dataset length : {}".format(self.triplet_train_dataset.len))
        print("Triplet Train dataset true length : {}".format(self.triplet_train_dataset.true_len))
        print("Triplet Val dataset length : {}".format(self.triplet_val_dataset.len))
        print("Triplet Val dataset true length : {}".format(self.triplet_val_dataset.true_len))
        print("Vicreg Train dataset length : {}".format(self.vicreg_train_dataset.len))
        print("Vicreg Train dataset true length : {}".format(self.vicreg_train_dataset.true_len))
        print("Vicreg Val dataset length : {}".format(self.vicreg_val_dataset.len))
        print("Vicreg Val dataset true length : {}".format(self.vicreg_val_dataset.true_len))

        
        
        
    def load(self):
        # idk what data statistics file is for            
        # load the train data
        train_file_roots = [pickle_files_root for pickle_files_root in self.data_config['train']]
        val_file_roots = [pickle_files_root for pickle_files_root in self.data_config['val']]
        triplet_loss_terrains = set(terrain for terrain in self.data_config['triplet_loss'])
        vicreg_loss_terrains = [terrain for terrain in self.data_config['vicreg_loss']]

        triplet_loss_train_data = []
        triplet_loss_val_data = []
        vicreg_loss_train_data = []
        vicreg_loss_val_data = []

        for string in train_file_roots:
            if string.split('/')[-2] in triplet_loss_terrains:
                triplet_loss_train_data.append(string)
            if string.split('/')[-2] in vicreg_loss_terrains:
                vicreg_loss_train_data.append(string)
        for string in val_file_roots:
            if string.split('/')[-2] in triplet_loss_terrains:
                triplet_loss_val_data.append(string)
            if string.split('/')[-2] in vicreg_loss_terrains:
                vicreg_loss_val_data.append(string)

        print( 'data config path: ', self.data_config_path)

        # check if the data_statistics.pkl file exists
        if os.path.exists(self.data_config_path + '/data_statistics.pkl'):
            cprint('Loading the mean and std from the data_statistics.pkl file', 'green')
            data_statistics = pickle.load(open(self.data_config_path + '/data_statistics.pkl', 'rb'))
            self.mean, self.std = data_statistics['mean'], data_statistics['std']
            self.min, self.max = data_statistics['min'], data_statistics['max']
            
        else:
            # find the mean and std of the train dataset
            cprint('data_statistics.pkl file not found!', 'yellow')
            cprint('Finding the mean and std of the train dataset', 'green')
            self.tmp_dataset = ConcatDataset([VicregDataset(vicreg_loss_train_data+vicreg_loss_val_data, incl_orientation=self.include_orientation_imu) for pickle_files_root in self.data_config['train']])
            self.tmp_dataloader = DataLoader(self.tmp_dataset, batch_size=128, num_workers=2, shuffle=False)
            cprint('the length of the tmp_dataloader is : {}'.format(len(self.tmp_dataloader)), 'green')
            # find the mean and std of the train dataset
            imu_data, leg_data, feet_data = [], [], []
            for _, _, imu, leg, feet, _, _ in tqdm(self.tmp_dataloader):
                imu_data.append(imu.cpu().numpy())
                leg_data.append(leg.cpu().numpy())
                feet_data.append(feet.cpu().numpy())
            imu_data = np.concatenate(imu_data, axis=0)
            leg_data = np.concatenate(leg_data, axis=0)
            feet_data = np.concatenate(feet_data, axis=0)
            print('imu_data.shape : ', imu_data.shape)
            print('leg_data.shape : ', leg_data.shape)
            print('feet_data.shape : ', feet_data.shape)
            # exit() # why?
            
            imu_data = imu_data.reshape(-1, imu_data.shape[-1])
            leg_data = leg_data.reshape(-1, leg_data.shape[-1])
            feet_data = feet_data.reshape(-1, feet_data.shape[-1])
            
            self.mean['imu'], self.std['imu'] = np.mean(imu_data, axis=0), np.std(imu_data, axis=0)
            self.min['imu'], self.max['imu'] = np.min(imu_data, axis=0), np.max(imu_data, axis=0)
            
            self.mean['leg'], self.std['leg'] = np.mean(leg_data, axis=0), np.std(leg_data, axis=0)
            self.min['leg'], self.max['leg'] = np.min(leg_data, axis=0), np.max(leg_data, axis=0)
            
            self.mean['feet'], self.std['feet'] = np.mean(feet_data, axis=0), np.std(feet_data, axis=0)
            self.min['feet'], self.max['feet'] = np.min(feet_data, axis=0), np.max(feet_data, axis=0)
            
            cprint('Mean : {}'.format(self.mean), 'green')
            cprint('Std : {}'.format(self.std), 'green')
            cprint('Min : {}'.format(self.min), 'green')
            cprint('Max : {}'.format(self.max), 'green')
            
            # save the mean and std
            cprint('Saving the mean, std, min, max to the data_statistics.pkl file', 'green')
            data_statistics = {'mean': self.mean, 'std': self.std, 'min': self.min, 'max': self.max}
            
            pickle.dump(data_statistics, open(self.data_config_path + '/data_statistics.pkl', 'wb'))
            
        # load the train data

        self.triplet_train_dataset = TripletDataset(triplet_loss_train_data, train=True)
        self.triplet_val_dataset = TripletDataset(triplet_loss_val_data)
        self.vicreg_train_dataset = VicregDataset(vicreg_loss_train_data, incl_orientation=self.include_orientation_imu, data_stats=data_statistics, train=True)
        self.vicreg_val_dataset = VicregDataset(vicreg_loss_val_data, incl_orientation=self.include_orientation_imu, data_stats=data_statistics)
    def train_dataloader(self):
        # return a zip of the two dataloaders
        triplet_dataloader = DataLoader(self.triplet_train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, 
                          drop_last= True if len(self.triplet_train_dataset) % self.batch_size != 0 else False,
                          pin_memory=True)
        vicreg_dataloader = DataLoader(self.vicreg_train_dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True,
                            drop_last= True if len(self.vicreg_train_dataset) % self.batch_size != 0 else False,
                            pin_memory=True)

        dataloader_length = len(list(zip(triplet_dataloader, vicreg_dataloader)))
        print("Length of train_dataloader:", dataloader_length)
        
        return zip(triplet_dataloader, vicreg_dataloader)

    def val_dataloader(self):
        # return a zip of the two dataloaders
        triplet_dataloader = DataLoader(self.triplet_val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False, 
                          drop_last= False,
                          pin_memory=True)
        vicreg_dataloader = DataLoader(self.vicreg_val_dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=False,
                            drop_last= False,
                            pin_memory=True)
        return zip(triplet_dataloader, vicreg_dataloader)

    def test_dataloader(self):
        # return a zip of the two dataloaders
        triplet_dataloader = DataLoader(self.triplet_val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False, 
                          drop_last= False,
                          pin_memory=False)
        vicreg_dataloader = DataLoader(self.vicreg_val_dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=False,
                            drop_last= False,
                            pin_memory=False)
        return zip(triplet_dataloader, vicreg_dataloader)

# run this from root directory
if __name__ == '__main__':
    dm = SplitTerrainDataModule('/home/luisamao/sterling/spot_data/split_dataset_configs/full_data.yaml', batch_size=128)
    dm.load()
    
    # Print the labels
    print(dm.triplet_train_dataset.labels)
    print(dm.triplet_train_dataset.labels)

    # print the lengths of each dataset
    print(len(dm.triplet_train_dataset))
    print(len(dm.vicreg_train_dataset))
    print("###################################")
    
    dataloader = dm.train_dataloader()
    # try getting an item
    # # start_time = time.time()
    # for triplet, vicreg in dataloader:
    #     print(triplet[0].shape)
    #     print(vicreg[0].shape)
    # # ...

    # Inside the loop where you iterate over the triplets
    for triplet, vicreg in dataloader:
        # triplet is a, p, n, label, idx
        # vicreg is patch1, patch2, imu, leg, feet, label, idx

        # triplet is [5, batch_size, num_channels, height, width]
        print(len(triplet))
        # exit()

        # triplet[0][0] is anchor, 1st in the batch. 3 x 64 x 64
        # Save the triplet as PNG files
        # print the shape of triplet[0][0]
        print("anchor image size", triplet[0][0].shape)
        print("transposed shape", triplet[0][0].numpy().transpose(1, 2, 0).shape)
        image_array = triplet[0][0].numpy().transpose(1, 2, 0)
        image_array = (image_array * 255).astype('uint8')
        anchor_img = Image.fromarray(image_array)
        # name the anchor with the label
        name = triplet[3][0]
        anchor_img.save(name+"_anchor.png")
        anchor_img = Image.fromarray((triplet[0][0].numpy().transpose(1, 2, 0)*255).astype('uint8'))
        positive_img = Image.fromarray((triplet[1][0].numpy().transpose(1, 2, 0)*255).astype('uint8'))
        negative_img = Image.fromarray((triplet[2][0].numpy().transpose(1, 2, 0)*255).astype('uint8'))


        positive_img.save("positive.png")
        anchor_img.save("anchor.png")
        negative_img.save("negative.png")

        print("Triplet saved as PNG files: positive.png, anchor.png, negative.png")
        break
            # end_time = time.time()
            # print("time: ", end_time - start_time)


