import numpy as np
from matplotlib import pyplot as plt

#mi = np.load('MI_DATA_SimpleCNN_relu_L4_B256_E1000_LR0.01_BINSIZE0.05_apr30_t1640.npy', allow_pickle=True).item()
mi = np.load('MI_DATA_SimpleCNN_tanh_L4_B256_E1000_LR0.01_BINSIZE0.05_apr01_t0753.npy', allow_pickle=True).item()
model = 'CNN'
activ = 'tanh'
epochs = np.arange(len(mi[0]['full'][:, 0]))
#plt.scatter(mi[0]['full'][:, 0], mi[0]['full'][:, 1], c=epochs, cmap='inferno', label='Mutual Information L1')
plt.scatter(mi[1]['full'][:, 0], mi[1]['full'][:, 1], c=epochs, cmap='inferno', label='Mutual Information L2')
plt.scatter(mi[2]['full'][:, 0], mi[2]['full'][:, 1], c=epochs, cmap='inferno', label='Mutual Information L1')
plt.scatter(mi[3]['full'][:, 0], mi[3]['full'][:, 1], c=epochs, cmap='inferno', label='Mutual Information L2')

plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.colorbar()
plt.title(f'MI plot for each layer [{model}, {activ}]')
plt.grid()
plt.savefig(f'MI_plot_{model}_{activ}_layers.png')
plt.show()


for j in range(4):
    plt.scatter(mi[0][f'F{j + 1}'][:, 0], mi[0][f'F{j + 1}'][:, 1], c=epochs, cmap='inferno',
                label=f'Mutual Information L1 F{j +1}')

plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.colorbar()
plt.grid()
plt.title(f'MI plot for each filter of L1 [{model}, {activ}]')
plt.savefig(f'MI_plot_{model}_{activ}_L1_filters.png')
plt.show()


for j in range(8):
    plt.scatter(mi[1][f'F{j + 1}'][:, 0], mi[1][f'F{j + 1}'][:, 1], c=epochs, cmap='inferno',
                label=f'Mutual Information L2 F{j +1}')

plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.colorbar()
plt.grid()
plt.title(f'MI plot for each filter of L2 [{model}, {activ}]')
plt.savefig(f'MI_plot_{model}_{activ}_L2_filters.png')
plt.show()


for j in range(8):
    plt.scatter(mi[2][f'F{j + 1}'][:, 0], mi[2][f'F{j + 1}'][:, 1], c=epochs, cmap='inferno',
                label=f'Mutual Information L3 F{j +1}')

plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.colorbar()
plt.grid()
plt.title(f'MI plot for each filter of L3 [{model}, {activ}]')
plt.savefig(f'MI_plot_{model}_{activ}_L3_filters.png')
plt.show()

for j in range(16):
    plt.scatter(mi[3][f'F{j + 1}'][:, 0], mi[3][f'F{j + 1}'][:, 1], c=epochs, cmap='inferno',
                label=f'Mutual Information L4 F{j +1}')

plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.colorbar()
plt.grid()
plt.title(f'MI plot for each filter of L4 [{model}, {activ}]')
plt.savefig(f'MI_plot_{model}_{activ}_L4_filters.png')
plt.show()