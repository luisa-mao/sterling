import sys
import os

# Add the top-level project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)
from train_combined_loss2 import NATURLRepresentationsModel
from combined_dataset import SplitTerrainDataModule
import torch
import torch.nn as nn
import torch.optim as optim

# Define your model
model = NATURLRepresentationsModel()

# Define your dataset and data loader
dm = SplitTerrainDataModule(data_config_path="spot_data/split_dataset_configs/full_data.yaml", batch_size=32)
data_loader = dm.train_dataloader()

# Define your loss function and optimizer
def criterion(triplet_out, vicreg_out):
    v_encoded_anch, v_encoded_pos, v_encoded_neg = triplet_out
    zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded = vicreg_out
    # compute viewpoint invariance vicreg loss
    loss_vpt_inv = model.vicreg_loss(zv1, zv2)
    # compute visual-inertial vicreg loss
    loss_vi = 0.5 * model.vicreg_loss(zv1, zi) + 0.5 * model.vicreg_loss(zv2, zi)
    v_loss = model.l1_coeff * loss_vpt_inv + (1.0-model.l1_coeff) * loss_vi

    # triplet loss
    # print (v_encoded_anch.shape, v_encoded_pos.shape, v_encoded_neg.shape)
    t_loss = model.triplet_loss(v_encoded_anch, v_encoded_pos, v_encoded_neg)
    loss = v_loss * 2/3 + t_loss * 1/3 # change coefficients here
    loss = v_loss
    return loss

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    running_loss = 0.0
    for triplet_in, vicreg_in in data_loader:
        print("here")
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        triplet_out, vicreg_out = model(triplet_in[:3], vicreg_in[:5])
        loss = criterion(triplet_out, vicreg_out)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

    # Print the average loss for the epoch
    # print(f"Epoch {epoch+1}: Loss = {running_loss / len(data_loader)}")
    print (f"End Epoch {epoch+1}")
    print(f"Epoch {epoch+1}: Loss = {running_loss}")
