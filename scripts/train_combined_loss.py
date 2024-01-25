#!/usr/bin/env python3

"""code to train the representation learning from the spot data"""
import sys
import os

# Add the top-level project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import torch
torch.multiprocessing.set_sharing_strategy('file_system') #https://github.com/pytorch/pytorch/issues/11201
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pickle
import glob
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F
import tensorboard as tb
import cv2
from tqdm import tqdm
from torch.nn import TripletMarginLoss

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime
import argparse
import yaml
import os
from sklearn import metrics
from scripts.combined_dataset import NATURLDataModule


from scipy.signal import periodogram
from scripts.models import ProprioceptionModel, VisualEncoderModel, VisualEncoderEfficientModel
from scripts.utils import process_feet_data, get_transforms

import albumentations as A
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


from termcolor import cprint

from scripts import cluster_jackal

# terrain_label = {
#     'cement': 0,
#     'pebble_pavement': 1,
#     'grass': 2,
#     'dark_tile': 3,
#     'bush': 4,
#     'asphalt': 5,
#     'marble_rock': 6,
#     'red_brick': 7, 
# }

terrain_label = {
    'asphalt': 0,
    'bush': 1,
    'concrete': 2,
    'grass': 3,
    'marble_rock': 4,
    'mulch': 5,
    'pebble_pavement': 6,
    'red_brick': 7,
    'yellow_brick': 8,
}

FEET_TOPIC_RATE = 24.0
LEG_TOPIC_RATE = 24.0
IMU_TOPIC_RATE = 200.0



class NATURLRepresentationsModel(pl.LightningModule):
    def __init__(self, lr=3e-4, latent_size=64, scale_loss=1.0/32, lambd=3.9e-6, weight_decay=1e-6, l1_coeff=0.5, rep_size=128):
        super(NATURLRepresentationsModel, self).__init__()
        
        self.save_hyperparameters(
            'lr',
            'latent_size',
            'scale_loss',
            'lambd',
            'weight_decay',
            'l1_coeff',
            'rep_size'
        )
        
        self.lr, self.latent_size, self.scale_loss, self.lambd, self.weight_decay = lr, latent_size, scale_loss, lambd, weight_decay
        self.l1_coeff = l1_coeff
        self.rep_size = rep_size
        
        # visual encoder architecture
        self.visual_encoder = VisualEncoderEfficientModel(latent_size=rep_size)
        # self.visual_encoder = VisualEncoderEfficientModel(latent_size=rep_size)
        
        self.proprioceptive_encoder = ProprioceptionModel(latent_size=rep_size)
        
        self.projector = nn.Sequential(
            nn.Linear(rep_size, latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size)
        )
        
        # coefficients for vicreg loss
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0
        
        self.max_acc = None

        self.triplet_loss = TripletMarginLoss(margin=1.0, p=2)
        
    
    def forward(self, patch1, patch2, neg_patch, inertial_data, leg, feet):
        v_encoded_1 = self.visual_encoder(patch1.float())
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2.float())
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)  
        v_encoded_neg = self.visual_encoder(neg_patch.float())
        v_encoded_neg = F.normalize(v_encoded_neg, dim=-1)      
        
        # i_encoded = self.inertial_encoder(inertial_data.float())
        i_encoded = self.proprioceptive_encoder(inertial_data.float(), leg.float(), feet.float())
        
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)
        
        return zv1, zv2, zi, v_encoded_1, v_encoded_2, v_encoded_neg, i_encoded
    
    def vicreg_loss(self, z1, z2):
        repr_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div_(z1.shape[1]) + self.off_diagonal(cov_y).pow_(2).sum().div_(z2.shape[1])
  
        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss
    
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def all_reduce(self, c):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)
            
    def training_step(self, batch, batch_idx):
        patch1, patch2, neg_patch, inertial, leg, feet, label, _ = batch
        
        # combine inertial and leg data
        # inertial = torch.cat((inertial, leg, feet), dim=-1)

        zv1, zv2, zi, v_encoded_1, v_encoded_2, v_encoded_neg, i_encoded = self.forward(patch1, patch2, neg_patch, inertial, leg, feet)
        
        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        
        loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi

        # triplet loss
        t_loss = self.triplet_loss(v_encoded_1, v_encoded_2, v_encoded_neg)
        loss = loss * 2/3 + t_loss * 1/3 # change coefficients here
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_vpt_inv', loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_vi', loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        patch1, patch2, neg_patch, inertial, leg, feet, label, _ = batch
        
        # combine inertial and leg data
        # inertial = torch.cat((inertial, leg, feet), dim=-1)
        
        zv1, zv2, zi, v_encoded_1, v_encoded_2, v_encoded_neg, i_encoded = self.forward(patch1, patch2, neg_patch, inertial, leg, feet)
        
        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        
        loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi
        # triplet loss
        t_loss = self.triplet_loss(v_encoded_1, v_encoded_2, v_encoded_neg)
        loss = loss * 2/3 + t_loss * 1/3 # change coefficients here
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_vpt_inv', loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_vi', loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        # return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        # return torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx =0):
        # save the batch data only every other epoch or during the last epoch
        if self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs-1:
            patch1, patch2, neg_patch, inertial, leg, feet, label, sampleidx = batch
            # combine inertial and leg data
            # inertial = torch.cat((inertial, leg, feet), dim=-1)
        
            with torch.no_grad():
                _, _, _, zv1, zv2, z_neg, zi = self.forward(patch1, patch2, neg_patch, inertial, leg, feet)
            zv1, zi = zv1.cpu(), zi.cpu()
            patch1 = patch1.cpu()
            label = np.asarray(label)
            sampleidx = sampleidx.cpu()
            
            if batch_idx == 0:
                self.visual_encoding = [zv1]
                self.inertial_encoding = [zi]
                self.label = label
                self.visual_patch = [patch1]
                self.sampleidx = [sampleidx]
            else:
                self.visual_encoding.append(zv1)
                self.inertial_encoding.append(zi)
                self.label = np.concatenate((self.label, label))
                self.visual_patch.append(patch1)
                self.sampleidx.append(sampleidx)
                
    # Find random groups of 25 images from each cluster
    def sample_clusters(self, clusters, elbow, vis_patch):

            # initialize
            dic = {}
            for a in range(elbow):
                dic[a] = []

            # For each cluster, find indexes of images in that cluster and extract 25 of them
            for i in range(elbow):

                idx = np.where(clusters == i)

                for _ in range(25):

                    # select correct patch
                    chosen = np.random.randint(low=0,high=len(idx[0]))
                    vp = vis_patch[idx[0][chosen], :, :, :]

                    # formatting for displayable image
                    vp = vp.cpu()
                    vp = vp.numpy()
                    vp= (vp * 255).astype(np.uint8)
                    vp = np.moveaxis(vp, 0, -1)

                    dic[i].append(vp)

            return dic
        
    # create and save 25 image grids for each cluster from dictionary image info
    # TODO: change file that images are saved to
    def img_clusters(self, dic, elbow, path_root="./models/"):

        for i in range(elbow):

            # initialize grid
            new_im = Image.new('RGB', (64*5,64*5))

            for j in range(25):

                vp = dic[i][j]
    
                # patch number to grid location
                h = int(j/5)
                w = j%5

                # format and paste individual patches to grid
                im = Image.fromarray(vp)
                im = im.convert('RGB')
                im.thumbnail((64,64))
                new_im.paste(im, (h*64,w*64))

            # save grid image
            new_im.save(path_root +"group"+str(i)+".png")
    
    def validate(self):
        print('Running validation...')
        dataset = self.trainer.datamodule.val_dataset
        self.visual_encoding, self.inertial_encoding, self.label, self.visual_patch, self.sampleidx = [], [], [], [], []
        # create dataloader for validation
        dataset = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
        
        for patch1, patch2, neg_patch, inertial, leg, feet, label, sampleidx in tqdm(dataset):
            # convert to torch tensors
            # patch1, patch2, inertial, leg, feet = torch.from_numpy(patch1), torch.from_numpy(patch2), torch.from_numpy(inertial), torch.from_numpy(leg), torch.from_numpy(feet)
            # move to device
            patch1, patch2, neg_patch, inertial, leg, feet = patch1.to(self.device), patch2.to(self.device), neg_patch.to(self.device), inertial.to(self.device), leg.to(self.device), feet.to(self.device)
            
            with torch.no_grad():
                # _, _, _, zv1, zv2, zi = self.forward(patch1.unsqueeze(0), patch2.unsqueeze(0), inertial.unsqueeze(0), leg.unsqueeze(0), feet.unsqueeze(0))
                _, _, _, zv1, zv2, z_neg, zi = self.forward(patch1, patch2, neg_patch, inertial, leg, feet)
                zv1, zi = zv1.cpu(), zi.cpu()
                patch1 = patch1.cpu()
                
            self.visual_patch.append(patch1)
            self.visual_encoding.append(zv1)
            self.inertial_encoding.append(zi)
            self.label.append(np.asarray(label))
            self.sampleidx.append(sampleidx)
        
        self.visual_patch = torch.cat(self.visual_patch, dim=0)
        self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
        self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
        self.sampleidx = torch.cat(self.sampleidx, dim=0)
        self.label = np.concatenate(self.label)
        
        # print('Visual Encoding Shape: {}'.format(self.visual_encoding.shape))
        # print('Inertial Encoding Shape: {}'.format(self.inertial_encoding.shape))
        # print('Visual Patch Shape: {}'.format(self.visual_patch.shape))
        # print('Sample Index Shape: {}'.format(self.sampleidx.shape))
    
    def on_validation_end(self):
        if (self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs-1) and torch.cuda.current_device() == 0:
            self.validate()
            
            # self.visual_patch = torch.cat(self.visual_patch, dim=0)
            # self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            # self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
            # self.sampleidx = torch.cat(self.sampleidx, dim=0)
            
            # cprint('Visual Encoding Shape: {}'.format(self.visual_encoding.shape), 'white', attrs=['bold'])
            
            # randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)
            
            # limit the number of samples to 2000
            ve = self.visual_encoding#[idx[:2000]]
            vi = self.inertial_encoding#[idx[:2000]]
            vis_patch = self.visual_patch#[idx[:2000]]
            ll = self.label#[idx[:2000]]
            
            data = torch.cat((ve, vi), dim=-1)
            
            # calculate and print accuracy
            cprint('finding accuracy...', 'yellow')
            accuracy, kmeanslabels, kmeanselbow, kmeansmodel = cluster_jackal.accuracy_naive(data, ll, label_types=list(terrain_label.keys()))
            fms, ari, chs = cluster_jackal.compute_fms_ari(data, ll, clusters=kmeanslabels, elbow=kmeanselbow, model=kmeansmodel)
            
            if not self.max_acc or accuracy > self.max_acc:
                self.max_acc = accuracy
                self.kmeanslabels, self.kmeanselbow, self.kmeansmodel = kmeanslabels, kmeanselbow, kmeansmodel
                self.vispatchsaved = torch.clone(vis_patch)
                self.sampleidxsaved = torch.clone(self.sampleidx)
                cprint('best model saved', 'green')
                
            # log k-means accurcay and projection for tensorboard visualization
            self.logger.experiment.add_scalar("K-means accuracy", accuracy, self.current_epoch)
            self.logger.experiment.add_scalar("Fowlkes-Mallows score", fms, self.current_epoch)
            self.logger.experiment.add_scalar("Adjusted Rand Index", ari, self.current_epoch)
            self.logger.experiment.add_scalar("Calinski-Harabasz Score", chs, self.current_epoch)
            self.logger.experiment.add_scalar("K-means elbow", self.kmeanselbow, self.current_epoch)
            
            # Save the cluster image grids on the final epoch only
            if self.current_epoch == self.trainer.max_epochs-1:
                path_root = "./models/acc_" + str(round(self.max_acc, 5)) + "_" + str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + "_" + "/"
                self.save_models(path_root)
                    
            
            if self.current_epoch % 10 == 0:
                self.logger.experiment.add_embedding(mat=data[idx[:2500]], label_img=vis_patch[idx[:2500]], global_step=self.current_epoch, metadata=ll[idx[:2500]], tag='visual_encoding')
            del self.visual_patch, self.visual_encoding, self.inertial_encoding, self.label
    
    def save_models(self, path_root='./models/'):
        cprint('saving models...', 'yellow', attrs=['bold'])
        if not os.path.exists(path_root): 
            cprint('creating directory: ' + path_root, 'yellow')
            os.makedirs(path_root)
        else:
            cprint('directory already exists: ' + path_root, 'red')
        
        dic = self.sample_clusters(self.kmeanslabels, self.kmeanselbow, self.vispatchsaved)
        self.img_clusters(dic, self.kmeanselbow, path_root=path_root)
        
        # ll = np.asarray([terrain_label[l] for l in ll])
        # dic = self.sample_clusters(ll, 8, vis_patch)
        # self.img_clusters(dic, 8)
        
        # save the kmeans model
        with open(os.path.join(path_root, 'kmeansmodel.pkl'), 'wb') as f:
            pickle.dump(self.kmeansmodel, f)
            cprint('kmeans model saved', 'green')
            
        # save the kmeans labels, true labels, and sample indices
        torch.save(self.kmeanslabels, os.path.join(path_root, 'kmeanslabels.pt'))
        torch.save(self.sampleidxsaved, os.path.join(path_root, 'sampleidx.pt'))
            
        # save the visual encoder
        torch.save(self.visual_encoder.state_dict(), os.path.join(path_root, 'visual_encoder.pt'))
        cprint('visual encoder saved', 'green')
        # save the proprioceptive encoder
        torch.save(self.proprioceptive_encoder.state_dict(), os.path.join(path_root, 'proprioceptive_encoder.pt'))
        cprint('proprioceptive encoder saved', 'green')
        cprint('All models successfully saved', 'green', attrs=['bold'])
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train representations using the NATURL framework')
    parser.add_argument('--batch_size', '-b', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--l1_coeff', type=float, default=0.5, metavar='L1C',
                        help='L1 loss coefficient (1)')
    parser.add_argument('--num_devices','-g', type=int, default=2, metavar='N',
                        help='number of devices to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=128, metavar='N',
                        help='Size of the common latent space (default: 128)')
    parser.add_argument('--save', type=int, default=0, metavar='N',
                        help='Whether to save the k means model and encoders at the end of the run')
    parser.add_argument('--imu_in_rep', type=int, default=1, metavar='N',
                        help='Whether to include the inertial data in the representation')
    parser.add_argument('--data_config_path', type=str, default='spot_data/data_config.yaml')
    parser.add_argument('--save_dir', type=str, default='combined_loss_logs/',
                        help='Directory to save logs and checkpoints')
    args = parser.parse_args()
    
    model = NATURLRepresentationsModel(lr=args.lr, latent_size=args.latent_size, l1_coeff=args.l1_coeff)
    dm = NATURLDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.save_dir)
    
    print("Training the representation learning model...")
    # trainer = pl.Trainer(devices=list(np.arange(args.num_devices, dtype=int)),
    trainer = pl.Trainer(
                         max_epochs=args.epochs,
                         log_every_n_steps=10,
                         strategy='ddp',
                         num_sanity_val_steps=0,
                         logger=tb_logger,
                         sync_batchnorm=True,
                         gradient_clip_val=100.0,
                         gradient_clip_algorithm='norm',
                         )

    # fit the model
    trainer.fit(model, dm)
    
    
    

