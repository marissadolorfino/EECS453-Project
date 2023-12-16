#!/usr/bin/python3

import sys
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import csv

molecule_data = sys.argv[1]

output_file = sys.argv[2]

tensor_transform = transforms.ToTensor()

class CustomDataset(Dataset):
    def __init__(self, train_data, train_targets, test_data, test_targets, transform=None):
        self.data = train_data
        self.targets = train_targets
        self.transform = transform
        self.test_data = test_data
        self.test_targets = test_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index], self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        return sample


# inputs are the molecule fingerprints
inputs = []
# targets are the bioactivity fingerprints
targets = []

# get inputs and targets
with open(molecule_data, 'r') as fingers:
    rows = fingers.readlines()
    rows.pop(0)

for row in rows:
    row = row.split(',')
    inp = np.array(list(row[2][2:-2]))
    targ = np.array(list(row[3][2:-3]))

    inp = inp.astype('int')
    targ = targ.astype('int')
    
    inputs.append(inp)
    targets.append(targ)

inputs = np.array(inputs)
targets = np.array(targets)    

# split into train and test data
inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.2, random_state=20)

# create molecular dataset
molecule_dataset = CustomDataset(inputs_train, targets_train, inputs_test, targets_test)

training_data = molecule_dataset.data
training_targets = molecule_dataset.targets

# convert to torch tensors
train_dataset = TensorDataset(torch.tensor(training_data), torch.tensor(training_targets))

# create a data loader
data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# length of molecular fingerprint: 1022 , length of bioactivity fingerprint: 496

# create a pytorch class for autoencoder and decoder
# autoencoder: 1022 --> 8 --> 496
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # build a linear encoder with Linear layers followed by Relu activation functions
        # 1022 --> 8
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1022, 820),
            torch.nn.ReLU(),
            torch.nn.Linear(820, 656),
            torch.nn.ReLU(),
            torch.nn.Linear(656, 525),
            torch.nn.ReLU(),
            torch.nn.Linear(525, 420),
            torch.nn.ReLU(),
            torch.nn.Linear(420, 210),
            torch.nn.ReLU(),
            torch.nn.Linear(210, 105),
            torch.nn.ReLU(),
            torch.nn.Linear(105, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8)
        )
         
        # build an linear decoder with Linear layer followed by Relu activation function
        # the Sigmoid activation function outputs the value between 0 and 1, good for binary values
        # 8 --> 496
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 105),
            torch.nn.ReLU(),
            torch.nn.Linear(105, 210),
            torch.nn.ReLU(),
            torch.nn.Linear(210, 420),
            torch.nn.ReLU(),
            torch.nn.Linear(420, 496),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x.float())
        decoded = self.decoder(encoded)
        threshold = 0.5
        binary_decoded = torch.where(decoded >= threshold, torch.tensor(1), torch.tensor(0))
        return binary_decoded.float()


# initialize model
molecule_model = AE()
 
for param in molecule_model.parameters():
    param.requires_grad = True

# train using MSE loss
MSE_loss = torch.nn.MSELoss()
 
# adam optimizer with lr = 0.1, l2 regularization weight = 1e-8
optimizer = torch.optim.Adam(molecule_model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

# training model
epochs = 10 
outputs = []
losses = []
epoch_index = 0

for epoch in range(epochs):
    epoch_index += 1
    
    batch_index = 0
    for (molecule_finger, true_bioactivity) in data_loader:
        batch_index += 1

        batch_losses = []
        
        # for each moleculer fingerprint and bioactivity fingerprint in the batch, predict the bioactivity, compute MSE  
        for i in range(len(molecule_finger)):
            single_molecule = molecule_finger[i, :]
            single_true_bioactivity = true_bioactivity[i, :]
            
            single_molecule = single_molecule.float()
            single_true_bioactivity = single_true_bioactivity.float()

            single_molecule.requires_grad = True
            single_true_bioactivity.requires_grad = True


            pred_bioactivity = molecule_model(single_molecule)
            single_loss = MSE_loss(pred_bioactivity, single_true_bioactivity)
            batch_losses.append(single_loss)
            
            outputs.append((epoch, single_true_bioactivity, pred_bioactivity))

        # average losses over batch
        loss = torch.mean(torch.stack(batch_losses))
        
        print('Epoch ', epoch_index, 'Batch ', batch_index, 'loss: ', loss) 

        # gradients set to zero, gradient computed and stored, .step() for parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # store batch loss for plotting
        loss = loss.detach()
        losses.append(loss)

print('training complete!')

# plot last 100 batch losses
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses[-100:])
plt.savefig('50000_more_layers_trainloss_plot.png')

# for testing on test set, need to convert datasets to tensors (torch.tensor())
# testing on test set:

test_inps = torch.tensor(molecule_dataset.test_data)
test_targs = torch.tensor(molecule_dataset.test_targets)

total_loss = []

outp_lines = [['molecule_finger', 'predicted_bioactivity', 'true_bioactivity']]

for i in range(len(test_inps)):
    if i < 50:
        test_molecule = test_inps[i, :]
        test_bio = test_targs[i, :]

        test_molecule = test_molecule.float()
        test_bio = test_bio.float()

        pred_bio = molecule_model(test_molecule)
        test_loss = MSE_loss(pred_bio, test_bio)
        total_loss.append(test_loss)
        
        # Convert tensors to lists of strings
        test_mole = [str(int(element)) for element in test_molecule.detach().cpu().numpy()]
        true_bio = [str(int(element)) for element in test_bio.detach().cpu().numpy()]
        pred = [str(int(element)) for element in pred_bio.detach().cpu().numpy()]

        # Combine the lists into strings
        test_mole_str = ''.join(test_mole)
        true_bio_str = ''.join(true_bio)
        pred_str = ''.join(pred)
        
        outp = [test_mole_str, pred_str, true_bio_str]
        outp_lines.append(outp)

    else:
        test_molecule = test_inps[i, :]
        test_bio = test_targs[i, :]

        test_molecule = test_molecule.float()
        test_bio = test_bio.float()

        pred_bio = molecule_model(test_molecule)
        test_loss = MSE_loss(pred_bio, test_bio)
        total_loss.append(test_loss)

print('Mean MSE on Test Set :', np.mean(total_loss))

plt.style.use('fivethirtyeight')
plt.xlabel('Test Molecule')
plt.ylabel('Loss')
plt.plot(total_loss)
plt.savefig('50000_more_layers_test_losses_plot.png')

with open(output_file, 'w') as out_file:
    writer = csv.writer(out_file, delimiter=',', lineterminator='\n')
    for out_line in outp_lines:
        writer.writerow(out_line)



