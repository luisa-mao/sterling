import sys
import os

import argparse


# Add the top-level project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from scripts.train_combined_loss2 import NATURLRepresentationsModel, terrain_label
from scripts.combined_dataset import SplitTerrainDataModule, TerrainDataModule
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np
from termcolor import cprint
from scripts import cluster_jackal
from datetime import datetime




class SterlingBaseline(NATURLRepresentationsModel):
    def __init__(self, lr=3e-4, latent_size=128, l1_coeff=0.5, devices=[0]):
        super().__init__(lr=lr, latent_size=latent_size, l1_coeff=l1_coeff, devices=devices, use_imu=True)
    def forward(self, vicreg_data):
        patch1, patch2, inertial_data, leg, feet, _, _ = vicreg_data

        v_encoded_1 = self.visual_encoder(patch1.float())
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2.float())
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)  
        i_encoded = self.proprioceptive_encoder(inertial_data.float(), leg.float(), feet.float())
        
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)

        vicreg_out = [zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded]
        
        return vicreg_out

    def calculate_loss(self, vicreg_out):
        zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded = vicreg_out
        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        v_loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi
        return v_loss, loss_vpt_inv, loss_vi

    def training_step(self, batch, batch_idx):
        vicreg_data = batch
        vicreg_out = self.forward(vicreg_data)
        loss, loss_vpt_inv, loss_vi = self.calculate_loss(vicreg_out)
        self.log('train_loss', loss)
        self.log('train_loss_vpt_inv', loss_vpt_inv)
        self.log('train_loss_vi', loss_vi)
        return loss

    def validation_step(self, batch, batch_idx):
        vicreg_data = batch
        vicreg_out = self.forward(vicreg_data)
        loss, loss_vpt_inv, loss_vi = self.calculate_loss(vicreg_out)
        self.log('val_loss', loss)
        self.log('val_loss_vpt_inv', loss_vpt_inv)
        self.log('val_loss_vi', loss_vi)
        return loss

    def validate(self):
        print('Running validation...')
        # dataset = self.trainer.datamodule.val_dataset
        self.visual_encoding, self.label, self.visual_patch, self.sampleidx = [], [], [], []
        val_dataloader = self.trainer.datamodule.val_dataloader()
        
        # for patch1, patch2, neg_patch, inertial, leg, feet, label, sampleidx in tqdm(dataset):
        for vicreg_data in val_dataloader:
            patch1, patch2, inertial_data, leg, feet, label, sampleidx = vicreg_data

            # move to device
            patch1, patch2, inertial, leg, feet = patch1.to(self.device), patch2.to(self.device), inertial.to(self.device), leg.to(self.device), feet.to(self.device)
            with torch.no_grad():
                vicreg_out = self.forward([patch1, patch2, inertial, leg, feet, label, sampleidx])
                zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded = vicreg_out
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
        print("validation end")

    # might be some weird things here
    # check terrain_labels when you actually run the experiment
    # i think lethal class should be in terrain_labels
    def on_validation_end(self):
        if (self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs-1) and torch.cuda.current_device() == self.devices[0]:
            self.validate()
            
            # randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)
            
            # limit the number of samples to 2000
            ve = self.visual_encoding#[idx[:2000]]
            # vi = self.inertial_encoding#[idx[:2000]]
            vis_patch = self.visual_patch#[idx[:2000]]
            ll = self.label#[idx[:2000]]
            
            # data = torch.cat((ve, vi), dim=-1)
            
            # calculate and print accuracy
            cprint('finding accuracy...', 'yellow')
            accuracy, kmeanslabels, kmeanselbow, kmeansmodel = cluster_jackal.accuracy_naive(ve, ll, label_types=list(terrain_label.keys()))
            fms, ari, chs = cluster_jackal.compute_fms_ari(ve, ll, clusters=kmeanslabels, elbow=kmeanselbow, model=kmeansmodel)
            
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
                self.logger.experiment.add_embedding(mat=ve[idx[:2500]], label_img=vis_patch[idx[:2500]], global_step=self.current_epoch, metadata=ll[idx[:2500]], tag='visual_encoding')
            del self.visual_patch, self.visual_encoding, self.label
    
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
    parser.add_argument('--data_config_path', type=str, default='spot_data/split_dataset_configs/full_data.yaml')
    parser.add_argument('--save_dir', type=str, default='combined_loss_logs/',
                        help='Directory to save logs and checkpoints')
    args = parser.parse_args()
    
    devices_to_use = [0,1,2,3,4]
    
    model = SterlingBaseline(lr=args.lr, latent_size=args.latent_size, l1_coeff=args.l1_coeff, devices = devices_to_use)
    tmp_dm = SplitTerrainDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)
    dm = TerrainDataModule(tmp_dm.combined_train_ds.vicreg_ds, tmp_dm.combined_val_ds.vicreg_ds, batch_size=args.batch_size)
    
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
                         devices=devices_to_use#args.num_devices
                         )

    # fit the model
    trainer.fit(model, dm)
    
