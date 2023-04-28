import os
import torch
import torchaudio
from utils import stereo_to_mono
#import soundfile as sf
#print(sf.available_formats())

data_dir = './data/fma_small'
output_dir = './data/spectrograms'
sampling_rate = 22_050
max_ms = 30_000
SUBSAMPLING = False


def spectro_gram(aud, n_mels=128, n_fft=1024, hop_len=512):
    sig, sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)


# create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# loop through all MP3 files in the data directory
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith('.mp3'):
            try:
                filepath = os.path.join(root, filename)#.replace("\\","/")


                sig, sr = torchaudio.load(filepath, format='mp3')

                sig = stereo_to_mono(sig)

                # resample to the desired sampling rate
                if sr != sampling_rate:
                    sig = torchaudio.transforms.Resample(sr, sampling_rate)(sig)

                if SUBSAMPLING:
                    subsample_length = sampling_rate * 5  # 5 seconds
                    overlap = int(subsample_length * 0.25)  # 25% overlap
                    subsamples = []
                    shift = subsample_length - overlap
                    for i in range(0, sig.size(0) - subsample_length + 1, shift):
                        subsample = sig[i:(i + subsample_length)]
                        subsamples.append(subsample)

                        if len(subsamples) == 7: break

                    # ignore samples that have less than 30s (it should not be the case but there is a 15s sample in validation dataset we dont know why)
                    if len(subsamples) != 7: continue

                    for idx, subsample in enumerate(subsamples):
                        # compute the spectrogram
                        spectro = spectro_gram(aud=(subsample, sampling_rate))

                        # save the spectrogram to the output directory
                        output_file = os.path.join(output_dir, filename[:-4] + f'_{idx}.pt')
                        torch.save(spectro, output_file)

                else:
                    # resize to a fixed length
                    sig_len = sig.shape[0]
                    print(sig_len)
                    max_len = sampling_rate // 1000 * max_ms
                    if sig_len > max_len:
                        sig = sig[:max_len]
                    elif sig_len < max_len:
                        pad_begin_len = random.randint(0, max_len - sig_len)
                        pad_end_len = max_len - sig_len - pad_begin_len
                        pad_begin = torch.zeros((num_rows, pad_begin_len))
                        pad_end = torch.zeros((num_rows, pad_end_len))
                        sig = torch.cat((pad_begin, sig, pad_end), 1)

                    # compute the spectrogram
                    spectro = spectro_gram(aud=(sig, sampling_rate))

                    # save the spectrogram to the output directory
                    output_file = os.path.join(output_dir, filename[:-4] + '.pt')
                    torch.save(spectro, output_file)
            except:

                continue
