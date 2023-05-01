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
from architectures import SimpleCNN, ResNet, SimpleCNN2
from simplebinmi import bin_calc_information2

#import kde
import kde_torch as kde

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


NUM_LABELS = 8
labelixs = {}
y = np.argmax(labels_onehot.to_numpy(), axis=1)
for i in range(NUM_LABELS):
    labelixs[i] = y == i

labelprobs = np.mean(labels_onehot, axis=0)

BATCH = 256
EPOCHS = 10
augment_prob = 0.8
labels_onehot_np = np.array(labels_onehot)

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
    #print(spec.size())
    #print(len(label))
    #print(ixs)
    #plot_spectrogram(spec[0])
    #print(spec.size(), ixs)
    #plot_spectrogram(spec[0])
    input_size = spec.size()[2]
    break

#plot MI I(X,T) for conv blocks

p_dropout = 0.3
#model = ResNet(FN=64, p_dropout=p_dropout)
#added a condition to allow to specify ReLU or tanh
model = SimpleCNN2(activation="ReLU")
#model = SimpleCNN()
model.to(device)

#summary(model, (1, 128, 1290))

#------------KDE functions
#nats to bits conversion factor
nats2bits = 1.0/np.log(2)
#upper/lower entropy estimates
noise_variance = 1e-3  # Added Gaussian noise variance
binsize = 0.07  # size of bins for binning method

# Functions to return upper and lower bounds on entropy of layer activity
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

mix_array_a1_u = []
mix_array_a2_u = []
mix_array_a3_u = []
mix_array_a4_u = []

mix_array_a1_l = []
mix_array_a2_l = []
mix_array_a3_l = []
mix_array_a4_l = []

miy_array_a1_u = []
miy_array_a2_u = []
miy_array_a3_u = []
miy_array_a4_u = []

miy_array_a1_l = []
miy_array_a2_l = []
miy_array_a3_l = []
miy_array_a4_l = []

h_array_a1_u = []
h_array_a2_u = []
h_array_a3_u = []
h_array_a4_u = []

h_array_a1_l = []
h_array_a2_l = []
h_array_a3_l = []
h_array_a4_l = []

activity1 = np.zeros((1000, 4, 10304))
activity2 = np.zeros((1000, 16, 2576))
activity3 = np.zeros((1000, 32, 648))
activity4 = np.zeros((1000, 64, 164))

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
        activity1[ixs] = a1.cpu().detach().numpy()
        activity2[ixs] = a2.cpu().detach().numpy()
        activity3[ixs] = a3.cpu().detach().numpy()
        activity4[ixs] = a4.cpu().detach().numpy()

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
    #------KDE estimates 1
    # Compute marginal entropies
    FN = 0
    h_upper = entropy_func_upper([activity1[:, FN, :], ])
    h_lower = entropy_func_lower([activity1[:, FN, :], ])
    # Layer activity given input. This is simply the entropy of the Gaussian noise
    hM_given_X = kde.kde_condentropy(activity1[:, FN, :], noise_variance)

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    hM_given_Y_lower = 0.
    for i in range(NUM_LABELS):
        hcond_upper = entropy_func_upper([activity1[labelixs[i], FN, :], ])
        hM_given_Y_upper += labelprobs[i] * hcond_upper
        hcond_lower = entropy_func_lower([activity1[labelixs[i], FN, :], ])
        hM_given_Y_lower += labelprobs[i] * hcond_lower
    #upper
    mix_array_a1_u.append(nats2bits * (h_upper - hM_given_X))
    miy_array_a1_u.append(nats2bits * (h_upper - hM_given_Y_upper))
    h_array_a1_u.append(nats2bits * h_upper * (1/10304))

    #lower
    mix_array_a1_l.append(nats2bits * (h_lower - hM_given_X))
    miy_array_a1_l.append(nats2bits * (h_lower - hM_given_Y_lower))
    h_array_a1_l.append(nats2bits * h_lower * (1/10304))

    #------KDE estimates 2
    # Compute marginal entropies
    FN = 0
    h_upper = entropy_func_upper([activity2[:, FN, :], ])
    h_lower = entropy_func_lower([activity2[:, FN, :], ])
    # Layer activity given input. This is simply the entropy of the Gaussian noise
    hM_given_X = kde.kde_condentropy(activity2[:, FN, :], noise_variance)

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    hM_given_Y_lower = 0.
    for i in range(NUM_LABELS):
        hcond_upper = entropy_func_upper([activity2[labelixs[i], FN, :], ])
        hM_given_Y_upper += labelprobs[i] * hcond_upper
        hcond_lower = entropy_func_lower([activity2[labelixs[i], FN, :], ])
        hM_given_Y_lower += labelprobs[i] * hcond_lower
    #upper
    mix_array_a2_u.append(nats2bits * (h_upper - hM_given_X))
    miy_array_a2_u.append(nats2bits * (h_upper - hM_given_Y_upper))
    h_array_a2_u.append(nats2bits * h_upper * (1/2576))

    #lower
    mix_array_a2_l.append(nats2bits * (h_lower - hM_given_X))
    miy_array_a2_l.append(nats2bits * (h_lower - hM_given_Y_lower))
    h_array_a2_l.append(nats2bits * h_lower * (1/2576))

    #------KDE estimates 3
    # Compute marginal entropies
    FN = 0
    h_upper = entropy_func_upper([activity3[:, FN, :], ])
    h_lower = entropy_func_lower([activity3[:, FN, :], ])
    # Layer activity given input. This is simply the entropy of the Gaussian noise
    hM_given_X = kde.kde_condentropy(activity3[:, FN, :], noise_variance)

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    hM_given_Y_lower = 0.
    for i in range(NUM_LABELS):
        hcond_upper = entropy_func_upper([activity3[labelixs[i], FN, :], ])
        hM_given_Y_upper += labelprobs[i] * hcond_upper
        hcond_lower = entropy_func_lower([activity3[labelixs[i], FN, :], ])
        hM_given_Y_lower += labelprobs[i] * hcond_lower
    #upper
    mix_array_a3_u.append(nats2bits * (h_upper - hM_given_X))
    miy_array_a3_u.append(nats2bits * (h_upper - hM_given_Y_upper))
    h_array_a3_u.append(nats2bits * h_upper * (1/648))

    #lower
    mix_array_a3_l.append(nats2bits * (h_lower - hM_given_X))
    miy_array_a3_l.append(nats2bits * (h_lower - hM_given_Y_lower))
    h_array_a3_l.append(nats2bits * h_lower * (1/648))

    #------KDE estimates 4
    # Compute marginal entropies
    FN = 0
    h_upper = entropy_func_upper([activity4[:, FN, :], ])
    h_lower = entropy_func_lower([activity4[:, FN, :], ])
    # Layer activity given input. This is simply the entropy of the Gaussian noise
    hM_given_X = kde.kde_condentropy(activity4[:, FN, :], noise_variance)

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    hM_given_Y_lower = 0.
    for i in range(NUM_LABELS):
        hcond_upper = entropy_func_upper([activity4[labelixs[i], FN, :], ])
        hM_given_Y_upper += labelprobs[i] * hcond_upper
        hcond_lower = entropy_func_lower([activity4[labelixs[i], FN, :], ])
        hM_given_Y_lower += labelprobs[i] * hcond_lower
    #upper
    mix_array_a4_u.append(nats2bits * (h_upper - hM_given_X))
    miy_array_a4_u.append(nats2bits * (h_upper - hM_given_Y_upper))
    h_array_a4_u.append(nats2bits * h_upper * (1/164))

    #lower
    mix_array_a4_l.append(nats2bits * (h_lower - hM_given_X))
    miy_array_a4_l.append(nats2bits * (h_lower - hM_given_Y_lower))
    h_array_a4_l.append(nats2bits * h_lower * (1/164))

mix_array_a1_u = np.array(mix_array_a1_u)
miy_array_a1_u = np.array(miy_array_a1_u)
h_array_a1_u = np.array(h_array_a1_u)
mix_array_a1_l = np.array(mix_array_a1_l)
miy_array_a1_l = np.array(miy_array_a1_l)
h_array_a1_l = np.array(h_array_a1_l)

np.save(timestamp + '_MIux_a1', mix_array_a1_u)
np.save(timestamp + '_MIuy_a1', miy_array_a1_u)
np.save(timestamp + '_MIuh_a1', h_array_a1_u)
np.save(timestamp + '_MIlx_a1', mix_array_a1_l)
np.save(timestamp + '_MIly_a1', miy_array_a1_l)
np.save(timestamp + '_MIlh_a1', h_array_a1_l)

mix_array_a2_u = np.array(mix_array_a2_u)
miy_array_a2_u = np.array(miy_array_a2_u)
h_array_a2_u = np.array(h_array_a2_u)
mix_array_a2_l = np.array(mix_array_a2_l)
miy_array_a2_l = np.array(miy_array_a2_l)
h_array_a2_l = np.array(h_array_a2_l)

np.save(timestamp + '_MIux_a2', mix_array_a2_u)
np.save(timestamp + '_MIuy_a2', miy_array_a2_u)
np.save(timestamp + '_MIuh_a2', h_array_a2_u)
np.save(timestamp + '_MIlx_a2', mix_array_a2_l)
np.save(timestamp + '_MIly_a2', miy_array_a2_l)
np.save(timestamp + '_MIlh_a2', h_array_a2_l)

mix_array_a3_u = np.array(mix_array_a3_u)
miy_array_a3_u = np.array(miy_array_a3_u)
h_array_a3_u = np.array(h_array_a3_u)
mix_array_a3_l = np.array(mix_array_a3_l)
miy_array_a3_l = np.array(miy_array_a3_l)
h_array_a3_l = np.array(h_array_a3_l)

np.save(timestamp + '_MIux_a3', mix_array_a3_u)
np.save(timestamp + '_MIuy_a3', miy_array_a3_u)
np.save(timestamp + '_MIuh_a3', h_array_a3_u)
np.save(timestamp + '_MIlx_a3', mix_array_a3_l)
np.save(timestamp + '_MIly_a3', miy_array_a3_l)
np.save(timestamp + '_MIlh_a3', h_array_a3_l)

mix_array_a4_u = np.array(mix_array_a4_u)
miy_array_a4_u = np.array(miy_array_a4_u)
h_array_a4_u = np.array(h_array_a4_u)
mix_array_a4_l = np.array(mix_array_a4_l)
miy_array_a4_l = np.array(miy_array_a4_l)
h_array_a4_l = np.array(h_array_a4_l)

np.save(timestamp + '_MIux_a4', mix_array_a4_u)
np.save(timestamp + '_MIuy_a4', miy_array_a4_u)
np.save(timestamp + '_MIuh_a4', h_array_a4_u)
np.save(timestamp + '_MIlx_a4', mix_array_a4_l)
np.save(timestamp + '_MIly_a4', miy_array_a4_l)
np.save(timestamp + '_MIlh_a4', h_array_a4_l)

#plot LOSS & ACCURACY

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_val, label='Validation loss')
plt.plot(loss_tr, label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.title('Loss vs Epochs')

plt.subplot(1, 2, 2)
plt.plot(acc_val, label='Validation accuracy')
plt.plot(acc_tr, label='Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.title('Accuracy vs Epochs')
plt.show()

plt.savefig('loss_acc.pdf')
#----------------------------------------------------------
#plot scatter UPPER
epochs = list(range(EPOCHS))
t = np.arange(len(mix_array_a1_u[:]))
plt.plot(mix_array_a1_u[:], miy_array_a1_u[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a1_u[:], miy_array_a1_u[:], c=t, cmap='inferno', label='Mutual Information Conv L1', zorder=2)
plt.plot(mix_array_a2_u[:], miy_array_a2_u[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a2_u[:], miy_array_a2_u[:], c=t, cmap='inferno', label='Mutual Information Conv L2', zorder=2)
plt.plot(mix_array_a3_u[:], miy_array_a3_u[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a3_u[:], miy_array_a3_u[:], c=t, cmap='inferno', label='Mutual Information Conv L3', zorder=2)
plt.plot(mix_array_a4_u[:], miy_array_a4_u[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a4_u[:], miy_array_a4_u[:], c=t, cmap='inferno', label='Mutual Information Conv L4', zorder=2)
plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.grid()
#plt.legend()
plt.colorbar()
plt.savefig('mi_xy_u.pdf')
plt.show()

#plot MI bar UPPER
plt.plot(epochs, h_array_a1_u, label='Entropy L1')
plt.plot(epochs, h_array_a2_u, label='Entropy L2')
plt.plot(epochs, h_array_a3_u, label='Entropy L3')
plt.plot(epochs, h_array_a4_u, label='Entropy L4')

plt.xlabel('Epochs')
plt.ylabel('Entropy(T)')
plt.grid()
plt.legend()
plt.savefig('h_u_epo.pdf')
plt.show()

#plot scatter LOWER
plt.plot(mix_array_a1_l[:], miy_array_a1_l[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a1_l[:], miy_array_a1_l[:], c=t, cmap='inferno', label='Mutual Information Conv L1', zorder=2)
plt.plot(mix_array_a2_l[:], miy_array_a2_l[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a2_l[:], miy_array_a2_l[:], c=t, cmap='inferno', label='Mutual Information Conv L2', zorder=2)
plt.plot(mix_array_a3_l[:], miy_array_a3_l[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a3_l[:], miy_array_a3_l[:], c=t, cmap='inferno', label='Mutual Information Conv L3', zorder=2)
plt.plot(mix_array_a4_l[:], miy_array_a4_l[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a4_l[:], miy_array_a4_l[:], c=t, cmap='inferno', label='Mutual Information Conv L4', zorder=2)
plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.grid()
#plt.legend()
plt.colorbar()
plt.savefig('mi_xy_l.pdf')
plt.show()

#plot h bar LOWER
plt.plot(epochs, h_array_a1_l, label='Entropy L1')
plt.plot(epochs, h_array_a2_l, label='Entropy L2')
plt.plot(epochs, h_array_a3_l, label='Entropy L3')
plt.plot(epochs, h_array_a4_l, label='Entropy L4')

plt.xlabel('Epochs')
plt.ylabel('Entropy(T)')
plt.grid()
plt.legend()
plt.savefig('h_l_epo.pdf')
plt.show()
#-------------------------------------------------------------------
#plot u/l MI vs epochs
#h = [nats2bits * np.log(10304) / np.log(8)] * len(epochs)
#plt.plot(epochs, h, linestyle='dashed')#upper bound on the mutual information
plt.plot(epochs, mix_array_a1_u, label='Upper Entropy')
plt.plot(epochs, mix_array_a1_l, label='Lower Entropy')
plt.xlabel('Epochs')
plt.ylabel('I(X,T)')
plt.title('Layer 1 Mutual Info (KDE)')
plt.grid()
plt.legend()
plt.savefig('mi_1_ul_epo.pdf')
plt.show()

#h = [nats2bits * np.log(2576) / np.log(8)] * len(epochs)
#plt.plot(epochs, h, linestyle='dashed')
plt.plot(epochs, mix_array_a2_u, label='Upper Entropy')
plt.plot(epochs, mix_array_a2_l, label='Lower Entropy')
plt.xlabel('Epochs')
plt.ylabel('I(X,T)')
plt.title('Layer 2 Mutual Info (KDE)')
plt.grid()
plt.legend()
plt.savefig('mi_2_ul_epo.pdf')
plt.show()

#h = [nats2bits * np.log(648) / np.log(8)] * len(epochs)
#plt.plot(epochs, h, linestyle='dashed')
plt.plot(epochs, mix_array_a3_u, label='Upper Entropy')
plt.plot(epochs, mix_array_a3_l, label='Lower Entropy')
plt.xlabel('Epochs')
plt.ylabel('I(X,T)')
plt.title('Layer 3 Upper Mutual Info (KDE)')
plt.grid()
plt.legend()
plt.savefig('mi_3_ul_epo.pdf')
plt.show()

#h = [nats2bits * np.log(164) / np.log(8)] * len(epochs)
#plt.plot(epochs, h, linestyle='dashed')
plt.plot(epochs, mix_array_a4_u, label='Upper Entropy')
plt.plot(epochs, mix_array_a4_l, label='Lower Entropy')
plt.xlabel('Epochs')
plt.ylabel('I(X,T)')
plt.title('Layer 4 Upper Mutual Info (KDE)')
plt.grid()
plt.legend()
plt.savefig('mi_4_ul_epo.pdf')
plt.show()
