import os
import torch
import torchaudio
from torch.utils.data import Dataset


class FMA2D_spec(Dataset):
    def __init__(self, data_dir, track_ids, labels_onehot, transforms=True, augment_prob=0.5, max_mask_pct=0.3, n_freq_masks=2,
                 n_time_masks=2):
        self.data_dir = data_dir
        self.filenames = os.listdir(data_dir)
        self.track_ids = track_ids
        self.labels_onehot = labels_onehot
        self.transforms = transforms
        self.augment_prob = augment_prob
        self.max_mask_pct = max_mask_pct
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __getitem__(self, index):
        tid = self.track_ids[index]
        # load the spectrogram data
        spec_path = os.path.join('./data/spectrograms/' + "{:06d}".format(tid) + '.pt')
        try:
            spec = torch.load(spec_path)
        except Exception as e:
            return self.__getitem__(index + 1)
        if self.transforms is True and torch.rand(1) < self.augment_prob:
            spec = self.spectro_augment(spec)
        # get label
        label = torch.from_numpy(self.labels_onehot.loc[tid].values).float()
        return spec, label, index

    def __len__(self):
        return len(self.track_ids)

    def spectro_augment(self, spec):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = self.max_mask_pct * n_mels
        for _ in range(self.n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = self.max_mask_pct * n_steps
        for _ in range(self.n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec