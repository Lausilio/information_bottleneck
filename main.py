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
#from architectures import SimpleCNN2, ResNet
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
EPOCHS = 5
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
model = SimpleCNN(activation="relu")
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

mi_array_a1 = []
mi_array_a2 = []
mi_array_a3 = []
mi_array_a4 = []

activity = np.zeros((1000, 4, 10304))
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
        #output, a1, a2 = model(spectrogram)
        output, a1, a2, a3, a4 = model(spectrogram)
        activity[ixs] = a1.cpu().detach().numpy()
        activity2[ixs] = a2.cpu().detach().numpy()
        activity3[ixs] = a3.cpu().detach().numpy()
        activity4[ixs] = a4.cpu().detach().numpy()

        loss = loss_fn(output, label)

        #cepochdata = defaultdict(list)

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
            val_output, a1, a2, _, _ = model(val_spectrogram)
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
    #------KDE estimates
    # Compute marginal entropies
    print(activity2.shape)
    FN = 0
    h_upper = entropy_func_upper([activity2[:, FN, :], ])
    h_lower = entropy_func_lower([activity2[:, FN, :], ])
    # Layer activity given input. This is simply the entropy of the Gaussian noise
    hM_given_X = kde.kde_condentropy(activity2[:, FN, :], noise_variance)

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    hM_given_Y_lower = 0.
    for i in range(NUM_LABELS):
        hcond_upper = entropy_func_upper([activity[labelixs[i], FN, :], ])
        hM_given_Y_upper += labelprobs[i] * hcond_upper
        hcond_lower = entropy_func_lower([activity[labelixs[i], FN, :], ])
        hM_given_Y_lower += labelprobs[i] * hcond_lower


    mi_array_a1.append(mi_a1)
    mi_array_a2.append(mi_a2)
    mi_array_a3.append(mi_a3)
    mi_array_a4.append(mi_a4)

mi_array_a1 = np.array(mi_array_a1)
mi_array_a2 = np.array(mi_array_a2)
mi_array_a3 = np.array(mi_array_a3)
mi_array_a4 = np.array(mi_array_a4)

np.save(timestamp + '_MI_a1', mi_array_a1)
np.save(timestamp + '_MI_a2', mi_array_a2)
np.save(timestamp + '_MI_a3', mi_array_a3)
np.save(timestamp + '_MI_a4', mi_array_a4)

t = np.arange(len(mi_array_a1))
plt.plot(t, mi_array_a1, label='MI Conv1')
plt.plot(t, mi_array_a2, label='MI Conv2')
plt.plot(t, mi_array_a3, label='MI Conv3')
plt.plot(t, mi_array_a4, label='MI Conv4')

plt.xlabel('Epochs')
plt.ylabel('MI/Entropy(T)')
plt.grid()
plt.legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('MI/Entropy(T) vs Epochs for Conv Blocks')

axs[0, 0].plot(t, mi_array_a1, label='MI Conv1', color='blue')
axs[0, 0].set_title('Conv1')
axs[0, 0].grid()
axs[0, 0].set(ylabel='MI/Entropy(T)')

axs[0, 1].plot(t, mi_array_a2, label='MI Conv2', color='orange')
axs[0, 1].set_title('Conv2')
axs[0, 1].grid()

axs[1, 0].plot(t, mi_array_a3, label='MI Conv3', color='green')
axs[1, 0].set_title('Conv3')
axs[1, 0].grid()
axs[1, 0].set(xlabel='Epochs', ylabel='MI/Entropy(T)')

axs[1, 1].plot(t, mi_array_a4, label='MI Conv4', color='red')
axs[1, 1].set_title('Conv4')
axs[1, 1].grid()
axs[1, 1].set(xlabel='Epochs')

plt.show()

torch.save(best_state_dict, model_name + f'_VAL{best_val_acc}_TRAIN{best_tr_acc}.pt')
print('Finished Training')




