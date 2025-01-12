import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torchaudio.transforms as tat
from torchaudio.transforms import FrequencyMasking, TimeMasking

from sklearn.metrics import accuracy_score
import gc
import wandb
import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime
from tqdm import tqdm

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

torch.cuda.empty_cache()
gc.collect()

# ARPABET PHONEME MAPPING
# DO NOT CHANGE

CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())


PHONEMES = CMUdict[:-2]
LABELS = ARPAbet[:-2]

root = '/home/yash/content/11785-f24-hw3p2'

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, partition):

        self.mfcc_dir = f'{root}/{partition}/mfcc'
        self.transcript_dir = f'{root}/{partition}/transcript'

        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.transcript_files = sorted(os.listdir(self.transcript_dir))
        assert len(self.mfcc_files) == len(self.transcript_files), "Mismatch in MFCC and transcript files."

        # Create phoneme-to-index mapping
        self.PHONEMES = PHONEMES
        self.phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(self.PHONEMES)}
        self.index_to_phoneme = {idx: phoneme for phoneme, idx in self.phoneme_to_index.items()}

        # Set the length of the dataset
        self.length = len(self.mfcc_files)

        # Preload all data into memory
        self.features = []
        self.labels = []
        
        for mfcc_file, transcript_file in zip(self.mfcc_files, self.transcript_files):
            # Load and append MFCC feature
            mfcc_path = os.path.join(self.mfcc_dir, mfcc_file)
            mfcc = np.load(mfcc_path)

            # Cepstral mean normalization
            mfcc -=  np.mean(mfcc, axis=0)
            mfcc /= np.std(mfcc, axis=0)

            self.features.append(mfcc)

            # Load, process transcript, and append label
            transcript_path = os.path.join(self.transcript_dir, transcript_file)
            transcript = np.load(transcript_path)[1:-1]
            
            # Filter out special tokens [SOS] and [EOS] before creating indices
            self.labels.append([self.phoneme_to_index[phoneme] for phoneme in transcript])


    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        '''
        Returns preloaded MFCC features and phoneme labels for the given index.
        '''
        mfcc = self.features[ind]
        transcript = self.labels[ind]
        return mfcc, transcript

    def collate_fn(self, batch):
        '''
        Pads sequences in the batch and returns padded features, padded labels,
        and the original sequence lengths for both features and labels.
        '''
        # Separate features and labels
        batch_mfcc = [item[0] for item in batch]
        batch_transcript = [torch.tensor(item[1], dtype=torch.long) for item in batch]

        batch_mfcc_pad = pad_sequence([torch.tensor(mfcc, dtype=torch.float32) for mfcc in batch_mfcc],
                batch_first=True,
                padding_value=0.0)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=-1)

        # Record lengths of original (unpadded) sequences
        lengths_mfcc = [mfcc.shape[0] for mfcc in batch_mfcc]
        lengths_transcript = [transcript.shape[0] for transcript in batch_transcript]

        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)


# Test Dataloader
class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        '''
        Initializes the test dataset by loading all MFCC features into memory.
        '''
        # Store directory path
        self.mfcc_dir = f'{root}/test-clean/mfcc'

        # List all .npy files in the mfcc directory
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))

        # Preload all MFCC data into memory
        self.features = []
        
        for mfcc_file in self.mfcc_files:
            # Load and append MFCC feature
            mfcc_path = os.path.join(self.mfcc_dir, mfcc_file)
            mfcc = np.load(mfcc_path)

            # Cepstral mean normalization
            mfcc -=  np.mean(mfcc, axis=0)
            mfcc /= np.std(mfcc, axis=0)

            self.features.append(mfcc)

        # Set the length of the dataset
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        '''
        Returns the preloaded MFCC features for the given index.
        '''
        return self.features[ind]
  

    def collate_fn(self, batch):
        '''
        Pads sequences in the batch and returns padded features and the original sequence lengths.
        '''
        # Pad sequences
        batch_mfcc = [torch.tensor(mfcc, dtype=torch.float32) for mfcc in batch]

        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0.0)

        # Record lengths of original (unpadded) sequences
        lengths_mfcc = [mfcc.shape[0] for mfcc in batch]

        return batch_mfcc_pad, torch.tensor(lengths_mfcc)


# Feel free to add more items here
config = {
    "beam_width" : 4,
    "lr"         : 2e-3,
    "epochs"     : 35,
    "batch_size" : 21
}

# You may pass this as a parameter to the dataset class above
# This will help modularize your implementation
# transforms = [
#     tat.TimeMasking(time_mask_param=30),
#     tat.FrequencyMasking(freq_mask_param=15)
# ] # set of tranformations

 
gc.collect()


# Create objects for the dataset class
train_data = AudioDataset(partition="train-clean-100")
val_data = AudioDataset(partition="dev-clean")
test_data = AudioDatasetTest() 

# Do NOT forget to pass in the collate function as parameter while creating the dataloader
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=train_data.collate_fn)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, collate_fn=val_data.collate_fn)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=test_data.collate_fn)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))


# sanity check
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break


torch.cuda.empty_cache()

class SEBlock1D(torch.nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // r, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y

class IdentityBlock1D(torch.nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock1D, self).__init__()
        
        filters1, filters2 = filters

        self.conv1 = nn.Conv1d(in_channels, filters1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(filters1)

        self.conv2 = nn.Conv1d(filters1, filters2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(filters2)

        self.se = SEBlock1D(filters2)
        self.relu = nn.ReLU()

        self.adjust_channels = None
        if in_channels != filters2:
            self.adjust_channels = nn.Conv1d(in_channels, filters2, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.se(x)

        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        x += identity
        x = self.relu(x)

        return x

class ResNet34(torch.nn.Module):
    def __init__(self, input_size, embed_size):
        super(ResNet34, self).__init__()

        self.embedding = nn.Conv1d(input_size, 192, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(192)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            IdentityBlock1D(192, [192, 192]),
            IdentityBlock1D(192, [192, 192]),
            IdentityBlock1D(192, [192, 192])
        )
        
        self.layer2 = nn.Sequential(
            IdentityBlock1D(192, [384, 384]),
            IdentityBlock1D(384, [384, 384]),
            IdentityBlock1D(384, [384, 384]),
            IdentityBlock1D(384, [384, 384])
        )

        self.layer3 = nn.Sequential(
            IdentityBlock1D(384, [768, 768]),
            IdentityBlock1D(768, [768, 768]),
            IdentityBlock1D(768, [768, 768]),
            IdentityBlock1D(768, [768, 768]),
            IdentityBlock1D(768, [768, 768]),
            IdentityBlock1D(768, [768, 768])
        )

        self.layer4 = nn.Sequential(
            IdentityBlock1D(768, [embed_size, embed_size]),
            IdentityBlock1D(embed_size, [embed_size, embed_size]),
            IdentityBlock1D(embed_size, [embed_size, embed_size])
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class PermuteBlock(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input?
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x_packed):
        x_packed, _ = self.blstm(x_packed)
        x, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        x, x_lens = self.trunc_reshape(x, x_lens)
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        return x_packed

    def trunc_reshape(self, x, x_lens):
        # If odd-length time steps, remove the last one
        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]
        
        batch_size, seq_len, feature_size = x.size()

        x = x.view(batch_size, seq_len // 2, feature_size * 2)

        x_lens = x_lens // 2

        return x, x_lens

class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size, num_layers = 2):
        super(Encoder, self).__init__()


        self.embedding = nn.Sequential(
            PermuteBlock(),
            ResNet34(input_size, encoder_hidden_size),
            PermuteBlock()
        )

        # Sequential pBLSTM layers
        self.pBLSTMs = nn.Sequential( pBLSTM(input_size=encoder_hidden_size, hidden_size=encoder_hidden_size),
            pBLSTM(input_size=encoder_hidden_size * 4, hidden_size=encoder_hidden_size))

    def forward(self, x, x_lens):
        # Apply embedding layer
        x = self.embedding(x)

        # Pack, pass through pBLSTM layers, unpack
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        x_out = self.pBLSTMs(x_packed)
        encoder_outputs, encoder_lens = pad_packed_sequence(x_out, batch_first=True)

        return encoder_outputs, encoder_lens


class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size= 41):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            PermuteBlock(), 
            nn.BatchNorm1d(embed_size), 
            PermuteBlock(),
            nn.Linear(embed_size, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, encoder_out):
        out = self.mlp(encoder_out)
        out = self.softmax(out)
        return out


class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size=512, output_size= len(PHONEMES)):
        super().__init__()

        self.augmentations = nn.Sequential(
            PermuteBlock(),
            FrequencyMasking(freq_mask_param=10),
            TimeMasking(time_mask_param=20),
            PermuteBlock()
        )
        self.encoder = Encoder(input_size, encoder_hidden_size=embed_size, num_layers=2)
        self.decoder = Decoder(embed_size * 4, output_size)

    def forward(self, x, lengths_x):

        if self.training:
            x = self.augmentations(x)
        encoder_out, encoder_lens = self.encoder(x, lengths_x)
        decoder_out = self.decoder(encoder_out)

        return decoder_out, encoder_lens


model = ASRModel(
    input_size=28,  # MFCC feature size
    embed_size=856,
    output_size=len(PHONEMES)
).to(device)
print(model)

summary(model, input_data=x.to(device), lengths_x=lx) 
 
# Initialize Loss Criterion, Optimizer, CTC Beam Decoder, Scheduler, Scaler (Mixed-Precision), etc.

# CTC Loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
# Refer to the handout for hints
criterion =  torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean') 

optimizer =  torch.optim.AdamW(model.parameters(), lr= config['lr'], weight_decay=1e-3)

# Declare the decoder. Use the CTC Beam Decoder to decode phonemes
# CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
decoder = CTCBeamDecoder(
    labels=PHONEMES,
    blank_id=0,
    beam_width=config['beam_width'],
    num_processes=4,
    log_probs_input=True
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)

# Mixed Precision, if you need it
scaler = torch.amp.GradScaler('cuda')


def decode_prediction(output, output_lens, decoder, PHONEME_MAP= LABELS):

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(output, seq_lens= output_lens) #lengths - list of lengths

    pred_strings = []

    for i in range(output_lens.shape[0]):
        # Get the best beam result for the current sample
        best_beam = beam_results[i][0][:out_lens[i][0]]
        
        # Map the indices to phonemes
        pred_string = ''.join([PHONEME_MAP[idx] for idx in best_beam])
        
        pred_strings.append(pred_string)

    return pred_strings

def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

    dist = 0
    batch_size = label.shape[0]

    pred_strings = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

    for i in range(batch_size):
        pred_string = pred_strings[i]
        label_string = ''.join([PHONEME_MAP[idx] for idx in label[i][:label_lens[i]]])
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size 
    return dist


# wandb.login(key="")


# run = wandb.init(
#     name = "reduced batch size and beam width", ## Wandb creates random run names if you skip this field
#     reinit = True, ### Allows reinitalizing runs when you re-run this cell
#     # run_id = ### Insert specific run id here if you want to resume a previous run
#     # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
#     project = "hw3p2-ablations", ### Project should be created in your wandb account
#     config = config ### Wandb Config for your run
# )



def train_model(model, train_loader, criterion, optimizer):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda'):
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        # Another couple things you need for FP16.
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


def validate_model(model, val_loader, decoder, phoneme_map= LABELS):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += float(loss)
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh, ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist


def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )

def load_model(path, model, metric= 'valid_acc', optimizer= None, scheduler= None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

    # if optimizer != None:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # if scheduler != None:
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # epoch   = checkpoint['epoch']
    # metric  = checkpoint[metric]

    # return [model, optimizer, scheduler, epoch, metric]


# This is for checkpointing, if you're doing it over multiple sessions

last_epoch_completed = 0
start = last_epoch_completed
end = config["epochs"]
best_lev_dist = float("inf") # if you're restarting from some checkpoint, use what you saw there.
epoch_model_path = '/home/yash/model/epoch.pth'
best_model_path = '/home/yash/model/best.pth'


torch.cuda.empty_cache()
gc.collect()

for epoch in range(0, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))

    curr_lr = float(optimizer.param_groups[0]['lr'])

    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_dist  = validate_model(model, val_loader, decoder, LABELS)
    scheduler.step()

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))


    wandb.log({
        'train_loss': train_loss,
        'valid_dist': valid_dist,
        'valid_loss': valid_loss,
        'lr'        : curr_lr
    })

    save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
    print("Saved epoch model")

    if valid_dist <= best_lev_dist:
        best_lev_dist = valid_dist
        save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
        print("Saved best model")
      # You may find it interesting to exlplore Wandb Artifcats to version your models
run.finish()

 
# Follow the steps below:
# 1. Create a new object for CTCBeamDecoder with larger (why?) number of beams
# 2. Get prediction string by decoding the results of the beam decoder

# load the best model
model = load_model(best_model_path, model)

TEST_BEAM_WIDTH = 5

test_decoder = CTCBeamDecoder(
    labels=LABELS,
    blank_id=0,
    beam_width=TEST_BEAM_WIDTH,
    num_processes=4,
    log_probs_input=True
)
results = []

model.eval()
print("Testing")
for data in tqdm(test_loader):

    x, lx   = data
    x       = x.to(device)

    with torch.no_grad():
        h, lh = model(x, lx)

    # call decode_prediction
    prediction_string = decode_prediction(h, lh, test_decoder, LABELS)
    results.extend(prediction_string)

    del x, lx, h, lh
    torch.cuda.empty_cache()


df = pd.DataFrame({"index": range(len(results)), "label": results})
df.to_csv('submission.csv', index = False)
print("Submission file created")

# kaggle competitions submit -c hw3p2-785-f24 -f submission.csv -m "I made it!"



