import os
import time
from collections import defaultdict

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

#import kde
import kde_torch as kde
#import keras.backend as K


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

labelprobs = np.mean(y, axis=0)

BATCH = 256
EPOCHS = 100
augment_prob = 0.8

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
    #print(spec.size(), ixs)
    #plot_spectrogram(spec[0])
    input_size = spec.size()[2]
    break

p_dropout = 0.3
#model = ResNet(FN=64, p_dropout=p_dropout)
model = SimpleCNN()
model.to(device)


#summary(model, (1, 128, 1290))

#------------KDE functions
# nats to bits conversion factor
nats2bits = 1.0/np.log(2)
#upper/lower entropy estimates
noise_variance = 1e-3  # Added Gaussian noise variance
binsize = 0.07  # size of bins for binning method

# Functions to return upper and lower bounds on entropy of layer activity
#Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder
#entropy_func_upper = torch.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
#entropy_func_lower = torch.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])
def entropy_func_upper(activity):
    return kde.entropy_estimator_kl(activity, noise_variance)
def entropy_func_lower(activity):
    return kde.entropy_estimator_bd(activity, noise_variance)
#------------------------

# Adam optimizer01
lr = 0.01
optimizer = torch.optim.Adam(model.parameters())

timestamp = time.strftime("apr%d_t%H%M", time.gmtime())
model_name = f"{model.name}_B{BATCH}_E{EPOCHS}_LR{lr}_pD{p_dropout}_A{augment_prob}_{timestamp}"

i = 0
running_loss = 0.0
best_val_loss = float('inf')  # initialize the best validation loss

# train the model
acc_tr = []
acc_val = []
loss_tr = []
loss_val = []
mi_array = []
mi2_array = []
mi3_array = []
mi4_array = []
activity = np.zeros((1000, 4, 10304))
activity2 = np.zeros((1000, 16, 2576))
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
        output, a1, a2 = model(spectrogram)
        activity[ixs] = a1.cpu().detach().numpy()
        activity2[ixs] = a2.cpu().detach().numpy()

        loss = loss_fn(output, label)

        cepochdata = defaultdict(list)

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
            val_output, a1, a2 = model(val_spectrogram)
            activity[ixs] = a1.cpu().detach().numpy()
            activity2[ixs] = a2.cpu().detach().numpy()
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
        '[{:.4f} min] Validation Loss: {:.4f} | Validation Accuracy: {:.4f} | Training Accuracy: {:.4f}'.format(t, loss,
                                                                                                                val_acc,
                                                                                                                tr_acc))
    #mi = bin_calc_information2(labelixs, activity[:, 0, :], 0.005)
    #print(activity[:, 0, :])
    #mi_array.append(mi)
#-----------KDE estimates
    # Compute marginal entropies
    h_upper = entropy_func_upper([activity[:, 0, :], ])#[0]
    h_lower = entropy_func_lower([activity[:, 0, :], ])#[0]

    # Layer activity given input. This is simply the entropy of the Gaussian noise
    hM_given_X = kde.kde_condentropy(activity[:, 0, :], noise_variance)

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    hM_given_Y_lower = 0.
    for i in range(NUM_LABELS):
        hcond_upper = entropy_func_upper([activity[labelixs[i], :], ])[0]
        hM_given_Y_upper += labelprobs[i] * hcond_upper
        hcond_lower = entropy_func_lower([activity[labelixs[i], :], ])[0]
        hM_given_Y_lower += labelprobs[i] * hcond_lower

    cepochdata['MI_XM_upper'].append(nats2bits * (h_upper - hM_given_X))
    cepochdata['MI_YM_upper'].append(nats2bits * (h_upper - hM_given_Y_upper))
    cepochdata['H_M_upper'].append(nats2bits * h_upper)
    pstr += 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])

    cepochdata['MI_XM_lower'].append(nats2bits * (h_lower - hM_given_X))
    cepochdata['MI_YM_lower'].append(nats2bits * (h_lower - hM_given_Y_lower))
    cepochdata['H_M_lower'].append(nats2bits * h_lower)
    pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])
#----------------------visualisation
mi_array_kde = np.array(cepochdata)
plt.scatter(mi_array_kde[:, 0], mi_array_kde[:, 1], label='Mutual Information L1')
plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.show()
#----------------------
#print(mi_array)
#mi_array = np.array(mi_array)
#np.save(timestamp, mi_array)

plt.plot(loss_val, label='Validation loss')
plt.plot(loss_tr, label='Training loss')
plt.show()

plt.plot(acc_val, label='Validation accuracy')
plt.plot(acc_tr, label='Training accuracy')
plt.show()

#plt.scatter(mi_array[:, 0], mi_array[:, 1], label='Mutual Information L1')
#plt.xlabel('I(X,T)')
#plt.ylabel('I(Y,T)')
#plt.show()

torch.save(best_state_dict, model_name + f'_VAL{best_val_acc}_TRAIN{best_tr_acc}.pt')
print('Finished Training')

