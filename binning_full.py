import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset
from torchsummary import summary

from utils import load
from dataloader import FMA2D_spec
from architectures import SimpleCNN, ResNet
from simplebinmi import bin_calc_information2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = './data/fma_small'

# download data first from these links:
# curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
# curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

tracks = load('./data/fma_metadata/tracks.csv')
subset = tracks.index[tracks['set', 'subset'] <= 'small']

tracks = tracks.loc[subset][:1000]
train = tracks.index[tracks['set', 'split'] == 'training']
val = tracks.index[tracks['set', 'split'] == 'validation']
test = tracks.index[tracks['set', 'split'] == 'test']

labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)
labels_onehot_np = np.array(labels_onehot)

NUM_LABELS = 8
labelixs = {}
y = np.argmax(labels_onehot_np, axis=1)
for i in range(NUM_LABELS):
    labelixs[i] = y == i


BATCH = 256
EPOCHS = 1000


# create a training dataset and dataloader
dataset_train = FMA2D_spec(DATA_DIR, train, labels_onehot, transforms=False)
dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH, shuffle=True)

# create a validation dataset and dataloader
dataset_valid = FMA2D_spec(DATA_DIR, val, labels_onehot, transforms=False)
val_dataloader = torch.utils.data.DataLoader(dataset_valid, batch_size=BATCH, shuffle=True)

# define the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()

# Lee 2017
# SGD optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)

from utils import plot_spectrogram
for spec, label, ixs in dataloader:
    input_size = spec.size()[2]
    break

p_dropout = 0.3
#model = ResNet(FN=64, p_dropout=p_dropout)
model = SimpleCNN()
model.to(device)


#summary(model, (1, 128, 1290))

# Adam optimizer01
lr = 0.01
optimizer = torch.optim.Adam(model.parameters())
BINSIZE = 0.05
ACTIV_FUNC = 'relu'
timestamp = time.strftime("apr%d_t%H%M", time.gmtime())

model_name = f"{model.name}_{ACTIV_FUNC}_L{model.layers_number}_B{BATCH}_E{EPOCHS}_LR{lr}_BINSIZE{BINSIZE}_{timestamp}"

i = 0
running_loss = 0.0
best_val_loss = float('inf')  # initialize the best validation loss

# train the model
acc_tr = []
acc_val = []
loss_tr = []
loss_val = []


mi_data = {0: {}, 1: {}, 2: {}, 3: {}}  # 0 for L1 and 1 for L2

activities = [np.zeros((1000, model.F1, 10304)),
              np.zeros((1000, model.F2, 2576)),
              np.zeros((1000, model.F3, 648)),
              np.zeros((1000, model.F4, 164))]

for layer_no in range(model.layers_number):
    mi_data[layer_no]['full'] = []
    for f in range(activities[layer_no].shape[1]):
        mi_data[layer_no][f'F{f + 1}'] = []

t0 = time.time()
prev_a = 0
for epoch in range(EPOCHS):
    # evaluate the model on the training dataset
    train_correct = 0
    train_total = 0
    for spectrogram, label, ixs in dataloader:
        model.train()
        label = label.to(device)
        train_label = torch.argmax(label, dim=1)

        # forward pass
        spectrogram = spectrogram.squeeze(0)
        spectrogram = spectrogram.unsqueeze(1)

        spectrogram = spectrogram.to(device)
        output, a1, a2, a3, a4 = model(spectrogram)
        activities[0][ixs] = a1.cpu().detach().numpy()
        activities[1][ixs] = a2.cpu().detach().numpy()
        activities[2][ixs] = a3.cpu().detach().numpy()
        activities[3][ixs] = a4.cpu().detach().numpy()

        loss = loss_fn(output, label)

        # backward pass
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate
        # scheduler.step(loss)

        _, train_predicted = torch.max(output.data, 1)
        train_total += train_label.size(0)
        train_correct += (train_predicted == train_label).sum().item()
        # print statistics
        i += 1
        running_loss += loss.item()

    loss = running_loss / len(dataloader)
    loss_tr.append(loss)
    print('[%d, %5d subsamples] Training loss: %.3f' % (epoch + 1, i * BATCH, loss))
    running_loss = 0
    # evaluate the model on the validation dataset
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()
    with torch.no_grad():
        for val_spectrogram, val_label, ixs in val_dataloader:
            val_label = val_label.to(device)
            val_label = torch.argmax(val_label, dim=1)

            val_spectrogram = val_spectrogram.squeeze(0)
            val_spectrogram = val_spectrogram.unsqueeze(1)
            val_spectrogram = val_spectrogram.to(device)
            val_output, a1, a2, a3, a4 = model(val_spectrogram)
            activities[0][ixs] = a1.cpu().detach().numpy()
            activities[1][ixs] = a2.cpu().detach().numpy()
            activities[2][ixs] = a3.cpu().detach().numpy()
            activities[3][ixs] = a4.cpu().detach().numpy()
            val_loss += loss_fn(val_output, val_label).item()
            _, val_predicted = torch.max(val_output.data, 1)
            val_total += val_label.size(0)
            val_correct += (val_predicted == val_label).sum().item()

    loss = val_loss / len(val_dataloader)
    loss_val.append(loss)
    val_acc = val_correct / val_total
    tr_acc = train_correct / train_total
    acc_tr.append(tr_acc)
    acc_val.append(val_acc)
    t1 = time.time()
    t = (t1 - t0) / 60
    # Save the model if the validation loss is the best seen so far
    if loss < best_val_loss:
        best_val_loss = loss
        best_val_acc = val_acc
        best_tr_acc = tr_acc
        best_state_dict = model.state_dict()
    print(
        '[EPOCH {}, {:.4f} min] Validation Loss: {:.4f} | Validation Accuracy: {:.4f} | Training Accuracy: {:.4f}'.format(epoch+1, t, loss,
                                                                                                                val_acc,
                                                                                   tr_acc))

    for idx, layer in enumerate(activities):
        layer_full = np.array([layer[fn].flatten() for fn in range(len(layer))])
        mi_data[idx]['full'].append(bin_calc_information2(labelixs, layer_full, BINSIZE))
        for f in range(layer.shape[1]):
            mi_data[idx][f'F{f+1}'].append(bin_calc_information2(labelixs, layer[:, f, :], BINSIZE))

for layer_no in range(model.layers_number):
    mi_data[layer_no]['full'] = np.array(mi_data[layer_no]['full'])
    for f in range(activities[layer_no].shape[1]):
        mi_data[layer_no][f'F{f + 1}'] = np.array(mi_data[layer_no][f'F{f + 1}'])

np.save(f'MI_DATA_{model_name}', mi_data)

plt.plot(loss_val, label='Validation loss')
plt.plot(loss_tr, label='Training loss')
plt.show()

plt.plot(acc_val, label='Validation accuracy')
plt.plot(acc_tr, label='Training accuracy')
plt.show()

t = np.arange(len(mi_data[0]['F1'][:, 0]))
plt.scatter(mi_data[0]['F1'][:, 0], mi_data[0]['F1'][:, 1], c=t, cmap='inferno', label='Mutual Information Conv1 L1')
plt.scatter(mi_data[0]['F2'][:, 0], mi_data[0]['F2'][:, 1], c=t, cmap='inferno', label='Mutual Information Conv1 L1')
plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.grid()
plt.legend()
plt.colorbar()
plt.show()


