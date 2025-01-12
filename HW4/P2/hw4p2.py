import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as tat
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, Dataset, DataLoader
import gc
import os
from transformers import AutoTokenizer
import yaml
import math
from typing import Literal, List, Optional, Any, Dict, Tuple
import random
import zipfile
import datetime
from torchinfo import summary
import glob
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fftpack import dct
import seaborn as sns
import matplotlib.pyplot as plt
import json
import warnings
import shutil

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

gc.collect()
torch.cuda.empty_cache()

with open("config.yaml") as file:
    config = yaml.safe_load(file)


class CharTokenizer:
    """A wrapper around character tokenization to have a consistent interface with other tokeization strategies"""

    def __init__(self):
        # Define special tokens for end-of-sequence, padding, and unknown characters
        self.eos_token = "<|endoftext|>"  # Same as EOS_TOKEN
        self.pad_token = "<|padding|>"
        self.unk_token = "<|unknown|>"

        # Initialize vocabulary with uppercase alphabet characters and space
        characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")

        # Create vocabulary mapping
        self.vocab = {
            self.eos_token: 0,
            self.pad_token: 1,  # Same ID as EOS_TOKEN
            self.unk_token: 2,
        }

        for idx, char in enumerate(characters, start=3):
            self.vocab[char] = idx

        # Create an inverse mapping from IDs to characters for decoding
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Define token IDs for special tokens for easy access
        self.eos_token_id = self.vocab[self.eos_token]
        self.bos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

        self.vocab_size = len(self.vocab)

    def tokenize(self, data: str) -> List[str]:
        # Split input data into a list of characters for tokenization
        return [char for char in data]

    def encode(
        self, data: str, return_tensors: Optional[Literal["pt"]] = None
    ) -> List[int]:
        # Encode each character in data to its integer ID, using unk_token if character is not in vocab
        e = [self.vocab.get(char.upper(), self.unk_token) for char in data]
        # If specified, convert to PyTorch tensor format
        if return_tensors == "pt":
            return torch.tensor(e).unsqueeze(0)
        return e

    def decode(self, data: List[int]) -> str:
        # Decode list of token IDs back to string by mapping each ID to its character
        try:
            return "".join([self.inv_vocab.get(j) for j in data])
        except:
            # Handle decoding error by converting data to list, if it's a tensor
            data = data.cpu().tolist()
            return "".join([self.inv_vocab.get(j) for j in data])


class GTokenizer:

    def __init__(
        self, token_type: Literal["1k", "10k", "50k", "char"] = "char", logger=None
    ):

        self.token_type = token_type
        self.vocab, self.inv_vocab = None, None
        if token_type == "1k":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_1k", use_fast=False
            )
        elif token_type == "10k":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_10k", use_fast=False
            )
        elif token_type == "20k":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_20k", use_fast=False
            )
        elif token_type == "50k":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_50k", use_fast=False
            )
        elif token_type == "100k":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_100k", use_fast=False
            )
        elif token_type == "char":
            self.tokenizer = CharTokenizer()

        self.EOS_TOKEN = self.tokenizer.eos_token_id
        self.SOS_TOKEN = self.tokenizer.bos_token_id
        self.PAD_TOKEN = (
            self.tokenizer.convert_tokens_to_ids("<|padding|>")
            if self.token_type != "char"
            else self.tokenizer.pad_token_id
        )

        self.UNK_TOKEN = self.tokenizer.unk_token_id
        self.VOCAB_SIZE = self.tokenizer.vocab_size

        # Test tokenization methods to ensure everything is working correctly
        test_text = "HI DEEP LEARNERS"
        test_tok = self.tokenize(test_text)
        test_enc = self.encode(test_text)
        test_dec = self.decode(test_enc)

        print(f"[Tokenizer Loaded]: {token_type}")
        print(f"\tEOS_TOKEN:  {self.EOS_TOKEN}")
        print(f"\tSOS_TOKEN:  {self.SOS_TOKEN}")
        print(f"\tPAD_TOKEN:  {self.PAD_TOKEN}")
        print(f"\tUNK_TOKEN:  {self.UNK_TOKEN}")
        print(f"\tVOCAB_SIZE: {self.VOCAB_SIZE}")
        print("Examples:")
        print(
            f"\t[DECODE EOS, SOS, PAD, UNK] : {self.decode([self.EOS_TOKEN, self.SOS_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN])}"
        )
        print(f"\t[Tokenize HI DEEP LEARNERS] : {test_tok}")
        print(f"\t[Encode   HI DEEP LEARNERS] : {test_enc}")
        print(f"\t[Decode   HI DEEP LEARNERS] : {test_dec}")

    def tokenize(self, data: str) -> List[str]:
        return self.tokenizer.tokenize(data)

    def encode(self, data: str, return_tensors=False) -> List[int]:
        if return_tensors:
            return self.tokenizer.encode(data, return_tensors="pt")
        return self.tokenizer.encode(data)

    def decode(self, data: List[int]) -> str:
        return self.tokenizer.decode(data)


Tokenizer = GTokenizer(config["token_type"])


class SpeechDataset(Dataset):

    def __init__(
        self,
        partition: Literal["train-clean-100", "dev-clean", "test-clean"],
        config: dict,
        tokenizer: GTokenizer,
        isTrainPartition: bool,
    ):
        """
        Initialize the SpeechDataset.

        Args:
            partition (str): Partition name
            config (dict): Configuration dictionary for dataset settings.
            tokenizer (GTokenizer): Tokenizer class for encoding and decoding text data.
            isTrainPartition (bool): Flag indicating if this partition is for training.
        """

        # general: Get config values
        self.config = config
        self.root = self.config["root"]
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.EOS_TOKEN
        self.sos_token = tokenizer.SOS_TOKEN
        self.pad_token = tokenizer.PAD_TOKEN
        self.subset = self.config["subset"]
        self.feat_type = self.config["feat_type"]
        self.num_feats = self.config["num_feats"]
        self.norm = self.config["norm"]

        # paths | files
        self.fbank_dir = os.path.join(self.root, self.partition, "fbank")
        self.fbank_files = sorted(os.listdir(self.fbank_dir))
        subset = int(self.subset * len(self.fbank_files))
        self.fbank_files = sorted(os.listdir(self.fbank_dir))[:subset]

        if "test-clean" not in self.partition:
            self.text_dir = os.path.join(self.root, self.partition, "text")
            self.text_files = sorted(os.listdir(self.text_dir))
            self.text_files = sorted(os.listdir(self.text_dir))[:subset]
            assert len(self.fbank_files) == len(
                self.text_files
            ), "Number of fbank files and text files must be the same"

        self.length = len(self.fbank_files)
        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []

        for i in tqdm(
            range(len(self.fbank_files)),
            desc=f"Loading fbank and transcript data for {self.partition}",
        ):
            # load features
            feats = np.load(os.path.join(self.fbank_dir, self.fbank_files[i]))
            if self.feat_type == "mfcc":
                feats = self.fbank_to_mfcc(feats)

            if self.config["norm"] == "cepstral":
                feats = (feats - np.mean(feats, axis=0)) / (
                    np.std(feats, axis=0) + 1e-8
                )

            self.feats.append(feats[: self.num_feats, :])

            # load and encode transcripts
            # Why do we have two different types of targets?
            # How do we want our decoder to know the start of sequence <SOS> and end of sequence <EOS>?

            if "test-clean" not in self.partition:
                # Note: You dont have access to transcripts in dev_clean
                transcript = np.load(
                    os.path.join(self.text_dir, self.text_files[i])
                ).tolist()
                transcript = "".join(transcript)
                # Invoke our tokenizer to tokenize the string
                tokenized = self.tokenizer.encode(transcript)

                ## TODO-DONE: How will you use tokenized?

                transcripts_shifted = [self.tokenizer.SOS_TOKEN] + tokenized
                transcripts_golden = tokenized + [self.tokenizer.EOS_TOKEN]

                self.transcripts_shifted.append(transcripts_shifted)
                self.transcripts_golden.append(transcripts_golden)

        if "test-clean" not in self.partition:
            assert (
                len(self.feats)
                == len(self.transcripts_shifted)
                == len(self.transcripts_golden)
            )

        # precompute global stats for global mean and variance normalization
        self.global_mean, self.global_std = None, None
        if self.config["norm"] == "global_mvn":
            self.global_mean, self.global_std = self.compute_global_stats()

        # Torch Audio Transforms
        # TODO-DONE
        # time masking
        self.time_mask = tat.TimeMasking(
            time_mask_param=config["specaug_conf"]["time_mask_width_range"]
        )

        # frequency masking
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config["specaug_conf"]["freq_mask_width_range"]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.feats[idx])
        shifted_transcript, golden_transcript = None, None
        if "test-clean" not in self.partition:
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript = torch.LongTensor(self.transcripts_golden[idx])
        # Apply global mean and variance normalization if enabled
        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean.unsqueeze(1)) / (
                self.global_std.unsqueeze(1) + 1e-8
            )
        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch):
        # @NOTE: batch corresponds to output from __getitem__ for a minibatch

        """
        1.  Extract the features and labels from 'batch'.
        2.  We will additionally need to pad both features and labels,
            look at PyTorch's documentation for pad_sequence.
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lengths of features, and lengths of labels.

        """
        # Prepare batch (features)
        batch_feats = [
            item[0].T for item in batch
        ]  # TODO: position of feats do you return from get_item + transpose B x T x F
        lengths_feats = [
            i.shape[0] for i in batch_feats
        ]  # Lengths of each T x F sequence
        batch_feats_pad = pad_sequence(
            batch_feats, batch_first=True, padding_value=self.pad_token
        )  # Pad sequence

        if "test-clean" not in self.partition:
            batch_transcript = [item[1] for item in batch]  # TODO: # B x T
            batch_golden = [item[2] for item in batch]  # TODO # B x T
            lengths_transcript = [len(t) for t in batch_transcript]  # Lengths of each T
            batch_transcript_pad = pad_sequence(
                batch_transcript,
                batch_first=True,
                padding_value=self.tokenizer.PAD_TOKEN,
            )
            batch_golden_pad = pad_sequence(
                batch_golden, batch_first=True, padding_value=self.tokenizer.PAD_TOKEN
            )

        # TODO: do specaugment transforms
        if self.config["specaug"] and self.isTrainPartition:

            # transpose back to F x T to apply transforms
            batch_feats_pad = batch_feats_pad.transpose(1, 2)

            # shape should be B x num_feats x T
            assert batch_feats_pad.shape[1] == self.num_feats

            # freq_mask
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    batch_feats_pad = self.freq_mask(batch_feats_pad)

            # time mask
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    batch_feats_pad = self.time_mask(batch_feats_pad)

            # transpose back to T x F
            batch_feats_pad = batch_feats_pad.transpose(1, 2)
            # shape should be B x T x num_feats
            assert batch_feats_pad.shape[2] == self.num_feats

        # Return the following values:
        # padded features, padded shifted labels, padded golden labels, actual length of features, actual length of the shifted label
        if "test-clean" not in self.partition:
            return (
                batch_feats_pad,
                batch_transcript_pad,
                batch_golden_pad,
                torch.tensor(lengths_feats),
                torch.tensor(lengths_transcript),
            )
        else:
            return batch_feats_pad, None, None, torch.tensor(lengths_feats), None

    def fbank_to_mfcc(self, fbank):
        # Helper function that applies the dct to the filterbank features to concert them to mfccs
        mfcc = dct(fbank.T, type=2, axis=1, norm="ortho")
        return mfcc.T

    # Will be discussed in bootcamp
    def compute_global_stats(self):
        all_feats = np.concatenate([feat for feat in self.feats], axis=1)

        # Compute global mean and std
        global_mean = np.mean(all_feats, axis=1)  # Shape: (num_feats,)
        global_std = np.std(all_feats, axis=1) + 1e-20  # Shape: (num_feats,)

        print(
            f"Computed global mean: {global_mean.shape}, global_variance: {global_std.shape}"
        )
        return torch.FloatTensor(global_mean), torch.FloatTensor(global_std)


gc.collect()


class TextDataset(Dataset):
    def __init__(self, partition: str, config: dict, tokenizer: GTokenizer):
        """
        Initializes the TextDataset class, which loads and tokenizes transcript files.

        Args:
            partition (str): Subdirectory under root that specifies the data partition (e.g., 'train', 'test').
            config (dict): Configuration dictionary for dataset settings.
            tokenizer (GTokenizer): Tokenizer instance for encoding transcripts into token sequences.
        """

        # General attributes
        self.root = config["root"]
        self.subset = config["subset"]
        self.partition = partition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.EOS_TOKEN  # End of sequence token
        self.sos_token = tokenizer.SOS_TOKEN  # Start of sequence token
        self.pad_token = tokenizer.PAD_TOKEN  # Padding token

        # Paths and files
        self.text_dir = os.path.join(
            self.root, self.partition
        )  # Directory containing transcript files
        self.text_files = sorted(
            os.listdir(self.text_dir)
        )  # Sorted list of transcript files

        # Limit to subset of files if specified
        subset = int(self.subset * len(self.text_files))
        self.text_files = self.text_files[:subset]
        self.length = len(self.text_files)

        # Storage for encoded transcripts
        self.transcripts_shifted, self.transcripts_golden = [], []

        # Load and encode transcripts
        for file in tqdm(
            self.text_files, desc=f"Loading transcript data for {partition}"
        ):
            transcript = np.load(os.path.join(self.text_dir, file)).tolist()
            transcript = " ".join(transcript.split())  # Process text
            tokenized = self.tokenizer.encode(transcript)  # Tokenize transcript
            # Store shifted and golden versions of transcripts
            self.transcripts_shifted.append(np.array([self.eos_token] + tokenized))
            self.transcripts_golden.append(np.array(tokenized + [self.eos_token]))

    def __len__(self) -> int:
        """Returns the total number of transcripts in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Retrieves the shifted and golden version of the transcript at the specified index.

        Args:
            idx (int): Index of the transcript to retrieve.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: Shifted and golden version of the transcript.
        """
        shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
        golden_transcript = torch.LongTensor(self.transcripts_golden[idx])
        return shifted_transcript, golden_transcript

    def collate_fn(
        self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collates a batch of transcripts for model input, applying padding as needed.

        Args:
            batch (List[Tuple[torch.LongTensor, torch.LongTensor]]): Batch of (shifted, golden) transcripts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Padded shifted transcripts (batch_transcript_pad).
                - Padded golden transcripts (batch_golden_pad).
                - Lengths of shifted transcripts.
        """

        # Separate shifted and golden transcripts from batch
        batch_transcript = [i[0] for i in batch]  # B x T
        batch_golden = [i[1] for i in batch]  # B x T
        lengths_transcript = [len(i) for i in batch_transcript]

        # Pad sequences
        batch_transcript_pad = pad_sequence(
            batch_transcript, batch_first=True, padding_value=self.pad_token
        )
        batch_golden_pad = pad_sequence(
            batch_golden, batch_first=True, padding_value=self.pad_token
        )

        # Return padded sequences and lengths
        return batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_transcript)


train_dataset = SpeechDataset(
    partition=config["train_partition"],
    config=config,
    tokenizer=Tokenizer,
    isTrainPartition=True,
)


val_dataset = SpeechDataset(
    partition=config["val_partition"],
    config=config,
    tokenizer=Tokenizer,
    isTrainPartition=False,
)


test_dataset = SpeechDataset(
    partition=config["test_partition"],
    config=config,
    tokenizer=Tokenizer,
    isTrainPartition=False,
)


text_dataset = TextDataset(
    partition=config["unpaired_text_partition"],
    config=config,
    tokenizer=Tokenizer,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["NUM_WORKERS"],
    pin_memory=True,
    collate_fn=train_dataset.collate_fn,
)


val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=config["NUM_WORKERS"],
    pin_memory=True,
    collate_fn=val_dataset.collate_fn,
)


test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["NUM_WORKERS"],
    pin_memory=True,
    collate_fn=test_dataset.collate_fn,
)

# UNCOMMENT if pretraining decoder as LM
text_loader = DataLoader(
    dataset=text_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["NUM_WORKERS"],
    pin_memory=True,
    collate_fn=text_dataset.collate_fn,
)


def verify_dataset(dataloader, partition):
    """Compute the Maximum MFCC and Transcript sequence length in a dataset"""
    print("Loaded Path: ", partition)
    max_len_feat = 0
    max_len_t = 0  # To track the maximum length of transcripts

    # Iterate through the dataloader
    for batch in tqdm(dataloader, desc=f"Verifying {partition} Dataset"):
        try:
            x_pad, y_shifted_pad, y_golden_pad, x_len, y_len = batch

            # Update the maximum feat length
            len_x = x_pad.shape[1]
            if len_x > max_len_feat:
                max_len_feat = len_x

            # Update the maximum transcript length
            # transcript length is dim 1 of y_shifted_pad
            if y_shifted_pad is not None:
                len_y = y_shifted_pad.shape[1]
                if len_y > max_len_t:
                    max_len_t = len_y

        except Exception as e:
            # The text dataset has no transcripts
            y_shifted_pad, y_golden_pad, y_len = batch

            # Update the maximum transcript length
            # transcript length is dim 1 of y_shifted_pad
            len_y = y_shifted_pad.shape[1]
            if len_y > max_len_t:
                max_len_t = len_y

    print(f"Maximum Feat Length in Dataset       : {max_len_feat}")
    print(f"Maximum Transcript Length in Dataset : {max_len_t}")
    return max_len_feat, max_len_t


print("")
print("Paired Data Stats: ")
print(f"No. of Train Feats   : {train_dataset.__len__()}")
print(f"Batch Size           : {config['batch_size']}")
print(f"Train Batches        : {train_loader.__len__()}")
print(f"Val Batches          : {val_loader.__len__()}")
# print(f"Test Batches         : {test_loader.__len__()}")
print("")
print("Checking the Shapes of the Data --\n")
for batch in train_loader:
    (
        x_pad,
        y_shifted_pad,
        y_golden_pad,
        x_len,
        y_len,
    ) = batch
    print(f"x_pad shape:\t\t{x_pad.shape}")
    print(f"x_len shape:\t\t{x_len.shape}")

    if y_shifted_pad is not None and y_golden_pad is not None and y_len is not None:
        print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
        print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
        print(f"y_len shape:\t\t{y_len.shape}\n")
        # convert one transcript to text
        transcript = train_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
        print(f"Transcript Shifted: {transcript}")
        transcript = train_dataset.tokenizer.decode(y_golden_pad[0].tolist())
        print(f"Transcript Golden: {transcript}")
    break
print("")

# UNCOMMENT if pretraining decoder as LM
print("Unpaired Data Stats: ")
print(f"No. of text          : {text_dataset.__len__()}")
print(f"Batch Size           : {config['batch_size']}")
print(f"Train Batches        : {text_loader.__len__()}")
print("")
print("Checking the Shapes of the Data --\n")
for batch in text_loader:
    (
        y_shifted_pad,
        y_golden_pad,
        y_len,
    ) = batch
    print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
    print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
    print(f"y_len shape:\t\t{y_len.shape}\n")

    # convert one transcript to text
    transcript = text_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
    print(f"Transcript Shifted: {transcript}")
    transcript = text_dataset.tokenizer.decode(y_golden_pad[0].tolist())
    print(f"Transcript Golden: {transcript}")
    break
print("")
print("\n\nVerifying Datasets")
max_train_feat, max_train_transcript = verify_dataset(
    train_loader, config["train_partition"]
)
max_val_feat, max_val_transcript = verify_dataset(val_loader, config["val_partition"])
max_test_feat, max_test_transcript = verify_dataset(
    test_loader, config["test_partition"]
)
_, max_text_transcript = verify_dataset(text_loader, config["unpaired_text_partition"])

MAX_SPEECH_LEN = max(max_train_feat, max_val_feat, max_test_feat)
MAX_TRANS_LEN = max(max_train_transcript, max_val_transcript)
print(f"Maximum Feat. Length in Entire Dataset      : {MAX_SPEECH_LEN}")
print(f"Maximum Transcript Length in Entire Dataset : {MAX_TRANS_LEN}")
print("")
gc.collect()


def PadMask(padded_input, input_lengths=None, pad_idx=None):
    """Create a mask to identify non-padding positions.

    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: Optional, the actual lengths of each sequence before padding, shape (N,).
        pad_idx: Optional, the index used for padding tokens.

    Returns:
        A mask tensor with shape (N, T), where padding positions are marked with 1 and non-padding positions are marked with 0.
    """
    # If input is a 2D tensor (N, T), add an extra dimension
    if padded_input.dim() == 2:
        padded_input = padded_input.unsqueeze(-1)

    mask = torch.ones(
        padded_input.shape[:2], dtype=torch.bool, device=padded_input.device
    )

    if input_lengths is not None:
        N, T, _ = padded_input.shape
        for i in range(N):
            mask[i, : input_lengths[i]] = False
            mask[i, input_lengths[i] :] = True
    else:
        mask = padded_input.squeeze(-1) == pad_idx  # Shape (N, T)
    return mask


def CausalMask(input_tensor):
    """
    Create an attention mask for causal self-attention based on input lengths.

    Args:
        input_tensor (torch.Tensor): The input tensor of shape (N, T, *).

    Returns:
        attn_mask (torch.Tensor): The causal self-attention mask of shape (T, T)
    """
    T = input_tensor.shape[1]  # Sequence length
    attn_mask = torch.zeros(T, T, dtype=torch.bool, device=input_tensor.device)
    # Shape (T, T)

    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=input_tensor.device), diagonal=1
    )
    # Lower triangular matrix

    attn_mask = attn_mask | causal_mask

    return attn_mask


class PositionalEncoding(torch.nn.Module):
    """Position Encoding from Attention Is All You Need Paper"""

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Initialize a tensor to hold the positional encodings
        pe = torch.zeros(max_len, d_model)

        # Create a tensor representing the positions (0 to max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for the sine and cosine functions
        # This term creates a series of values that decrease geometrically, used to generate varying frequencies for positional encodings
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the positional encodings tensor and make it a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class BiLSTMEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(BiLSTMEmbedding, self).__init__()
        self.bilstm = nn.LSTM(
            input_dim,
            output_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, x, x_len):
        """
        Args:
            x.    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # BiLSTM expects (batch_size, seq_len, input_dim)
        # Pack the padded sequence to avoid computing over padded tokens
        packed_input = pack_padded_sequence(
            x, x_len.cpu(), batch_first=True, enforce_sorted=False
        )
        # Pass through the BiLSTM
        packed_output, _ = self.bilstm(packed_input)
        # Unpack the sequence to restore the original padded shape
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output


### DO NOT MODIFY
class Conv2DSubsampling(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, dropout=0.0, time_stride=2, feature_stride=2
    ):
        """
        Conv2dSubsampling module that can selectively apply downsampling
        for time and feature dimensions, and calculate cumulative downsampling factor.
        Args:
            time_stride (int): Stride along the time dimension for downsampling.
            feature_stride (int): Stride along the feature dimension for downsampling.
        """
        super(Conv2DSubsampling, self).__init__()

        # decompose to get effective stride across two layers
        tstride1, tstride2 = self.closest_factors(time_stride)
        fstride1, fstride2 = self.closest_factors(feature_stride)

        self.feature_stride = feature_stride
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, kernel_size=3, stride=(tstride1, fstride1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=(tstride2, fstride2)
            ),
            torch.nn.ReLU(),
        )
        self.time_downsampling_factor = tstride1 * tstride2
        # Calculate output dimension for the linear layer
        conv_out_dim = (input_dim - (3 - 1) - 1) // fstride1 + 1
        conv_out_dim = (conv_out_dim - (3 - 1) - 1) // fstride2 + 1
        conv_out_dim = output_dim * conv_out_dim
        self.out = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, output_dim), torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            x_mask (torch.Tensor): Optional mask for the input tensor.

        Returns:
            torch.Tensor: Downsampled output of shape (batch_size, new_seq_len, output_dim).
        """
        x = x.unsqueeze(1)  # Add a channel dimension for Conv2D
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x

    def closest_factors(self, n):
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        # Return the factor pair
        return max(factor, n // factor), min(factor, n // factor)


class SpeechEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, time_stride, feature_stride, dropout):
        super(SpeechEmbedding, self).__init__()

        self.cnn = Conv2DSubsampling(
            input_dim,
            output_dim,
            dropout=dropout,
            time_stride=time_stride,
            feature_stride=feature_stride,
        )
        self.blstm = BiLSTMEmbedding(output_dim, output_dim, dropout)
        self.time_downsampling_factor = self.cnn.time_downsampling_factor

    def forward(self, x, x_len, use_blstm: bool = False):
        """
        Args:
            x    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len // stride, output_dim)
        """
        # First, apply Conv2D subsampling
        x = self.cnn(x)
        # Adjust sequence length based on downsampling factor
        x_len = torch.ceil(x_len.float() / self.time_downsampling_factor).int()
        x_len = x_len.clamp(max=x.size(1))

        # Apply BiLSTM if requested
        if use_blstm:
            x = self.blstm(x, x_len)

        return x, x_len


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):

        super(EncoderLayer, self).__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask):
        # Step 1: Apply pre-normalization
        """TODO"""
        x_pre = self.pre_norm(x)

        # Step 2: Self-attention with with dropout, and with residual connection
        """ TODO """
        x1, _ = self.self_attn(
            query=x_pre, key=x_pre, value=x_pre, key_padding_mask=pad_mask
        )
        x_post = x_pre + self.dropout(x1)

        # Step 3: Apply normalization
        """ TODO """
        x_post = self.norm1(x_post)

        # Step 4: Apply Feed-Forward Network (FFN) with dropout, and residual connection
        """ TODO """
        x2 = self.ffn1(x_post)
        x_post = x_post + self.dropout(x2)

        # Step 5: Apply normalization after FFN
        """ TODO """
        x = self.norm2(x_post)

        return x, pad_mask


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        max_len,
        target_vocab_size,
        dropout=0.1,
    ):

        super(Encoder, self).__init__()

        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout).to(device)
                for _ in range(num_layers)
            ]
        )
        self.after_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, target_vocab_size)

    def forward(self, x, x_len):
        # Step 1: Create padding mask for inputs - ensure it's on the same device
        pad_mask = PadMask(x, x_len).to(device)

        # Step 2: Apply positional encoding - ensure positions tensor is on the same device
        x = self.pos_encoding(x)

        # Step 3: Apply dropout
        x = self.dropout(x)

        # Step 4: Pass through all encoder layers
        for layer in self.enc_layers:
            x, pad_mask = layer(x, pad_mask)

        # Step 5: Apply final normalization
        x = self.after_norm(x)

        # Step 6: Pass a branch through the CTC head
        x_ctc = self.ctc_head(x)

        # Step 7: Apply log softmax and permute
        x_ctc = x_ctc.log_softmax(2).permute(1, 0, 2)

        return x, x_len, x_ctc


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.mha1 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.identity = nn.Identity()
        self.pre_norm = nn.LayerNorm(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self, padded_targets, enc_output, pad_mask_enc, pad_mask_dec, slf_attn_mask
    ):
        x_pre = self.pre_norm(padded_targets)

        # Self-attention with pre-norm
        mha1_output, mha1_attn_weights = self.mha1(
            x_pre,
            x_pre,
            x_pre,
            attn_mask=slf_attn_mask,
            key_padding_mask=pad_mask_dec,
            need_weights=True,
        )
        mha1_output = self.dropout1(mha1_output)
        x1 = mha1_output + padded_targets
        x2 = self.layernorm1(x1)

        # Cross-attention with pre-norm
        if enc_output is not None:
            mha2_output, mha2_attn_weights = self.mha2(
                query=x2,
                key=enc_output,
                value=enc_output,
                key_padding_mask=pad_mask_enc,
                need_weights=True,
            )
        else:
            mha2_output = self.identity(padded_targets)
            mha2_attn_weights = torch.zeros_like(mha1_attn_weights)

        mha2_output = self.dropout2(mha2_output)
        mha2_output = mha2_output + x1
        mha2_output_post = self.layernorm2(mha2_output)

        # Feed-forward with pre-norm
        ffn_output = self.ffn(mha2_output_post)

        ffn_output = self.dropout3(ffn_output)

        ffn_output = ffn_output + mha2_output
        ffn_output = self.layernorm3(ffn_output)

        return ffn_output, mha1_attn_weights, mha2_attn_weights


class Decoder(torch.nn.Module):
    def __init__(
        self, num_layers, d_model, num_heads, d_ff, dropout, max_len, target_vocab_size
    ):

        super().__init__()

        self.max_len = max_len
        self.num_layers = num_layers
        self.num_heads = num_heads

        # use torch.nn.ModuleList() with list comprehension looping through num_layers
        # @NOTE: think about what stays constant per each DecoderLayer (how to call DecoderLayer)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.final_linear = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, padded_targets, target_lengths, enc_output, enc_input_lengths):
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(
                padded_input=padded_targets, input_lengths=target_lengths
            ).to(padded_targets.device)
        causal_mask = CausalMask(padded_targets).to(padded_targets.device)

        # Apply embedding
        padded_targets = self.target_embedding(padded_targets)

        # Apply positional encoding
        padded_targets = self.positional_encoding(padded_targets)

        # Create padding mask for encoder outputs
        pad_mask_enc = None
        if enc_output is not None:
            pad_mask_enc = PadMask(
                padded_input=enc_output, input_lengths=enc_input_lengths
            ).to(enc_output.device)

        # Pass through decoder layers
        runnint_att = {}
        for i in range(self.num_layers):
            (
                padded_targets,
                runnint_att[f"layer{i+1}_dec_self"],
                runnint_att[f"layer{i+1}_dec_cross"],
            ) = self.dec_layers[i](
                padded_targets,
                enc_output,
                pad_mask_enc,
                pad_mask_dec,
                causal_mask,
            )

        # Final projection
        seq_out = self.final_linear(padded_targets)

        return seq_out, runnint_att

    def recognize_greedy_search(self, enc_output, enc_input_lengths, tokenizer):
        """passes the encoder outputs and its corresponding lengths through autoregressive network
        @NOTE: You do not need to make changes to this method.
        """
        # start with the <SOS> token for each sequence in the batch
        batch_size = enc_output.size(0)
        target_seq = torch.full(
            (batch_size, 1), tokenizer.SOS_TOKEN, dtype=torch.long
        ).to(enc_output.device)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(enc_output.device)

        for _ in range(self.max_len):

            seq_out, runnint_att = self.forward(
                target_seq, None, enc_output, enc_input_lengths
            )
            logits = torch.nn.functional.log_softmax(seq_out[:, -1], dim=1)

            # selecting the token with the highest probability
            # @NOTE: this is the autoregressive nature of the network!
            # appending the token to the sequence
            # checking if <EOS> token is generated
            # or opration, if both or one of them is true store the value of the finished sequence in finished variable
            # end if all sequences have generated the EOS token
            next_token = logits.argmax(dim=-1).unsqueeze(1)
            target_seq = torch.cat([target_seq, next_token], dim=-1)
            eos_mask = next_token.squeeze(-1) == tokenizer.EOS_TOKEN
            finished |= eos_mask
            if finished.all():
                break

        # remove the initial <SOS> token and pad sequences to the same length
        target_seq = target_seq[:, 1:]
        max_length = target_seq.size(1)
        target_seq = torch.nn.functional.pad(
            target_seq, (0, self.max_len - max_length), value=tokenizer.PAD_TOKEN
        )

        return target_seq

    def recognize_beam_search(
        self, enc_output, enc_input_lengths, tokenizer, beam_width=5
    ):
        # TODO Beam Decoding
        batch_size = enc_output.size(0)
        device = enc_output.device

        beams = [
            [
                (
                    torch.tensor(
                        [tokenizer.SOS_TOKEN], device=device, dtype=torch.long
                    ),
                    0.0,
                )
            ]
            for _ in range(batch_size)
        ]

        completed_sequences = [[] for _ in range(batch_size)]

        for _ in range(self.max_len):

            all_candidates = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                if len(completed_sequences[i]) >= beam_width:
                    continue

                for seq, score in beams[i]:
                    if seq[-1] == tokenizer.EOS_TOKEN:
                        completed_sequences[i].append((seq, score))
                        continue

                    seq_out, _ = self.forward(
                        seq.unsqueeze(0),
                        None,
                        enc_output[i].unsqueeze(0),
                        enc_input_lengths[i].unsqueeze(0),
                    )
                    logits = torch.nn.functional.log_softmax(
                        seq_out[:, -1], dim=1
                    ).squeeze(0)

                    topk_probs, topk_tokens = logits.topk(beam_width)

                    for k in range(beam_width):
                        new_seq = torch.cat([seq, topk_tokens[k].unsqueeze(0)])
                        new_score = score + topk_probs[k].item()
                        all_candidates[i].append((new_seq, new_score))

            for i in range(batch_size):
                if len(completed_sequences[i]) < beam_width:
                    beams[i] = sorted(
                        all_candidates[i], key=lambda x: x[1], reverse=True
                    )[:beam_width]
                else:
                    beams[i] = []

        final_sequences = []
        for i in range(batch_size):

            if completed_sequences[i]:

                final_sequences.append(
                    sorted(completed_sequences[i], key=lambda x: x[1], reverse=True)[0][
                        0
                    ]
                )
            else:
                final_sequences.append(
                    sorted(beams[i], key=lambda x: x[1], reverse=True)[0][0]
                )

        final_sequences = [seq[1:] for seq in final_sequences]
        padded_sequences = torch.full(
            (batch_size, self.max_length),
            tokenizer.PAD_TOKEN,
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(final_sequences):
            padded_sequences[i, : seq.size(0)] = seq

        return padded_sequences


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        time_stride,
        feature_stride,
        embed_dropout,
        d_model,
        enc_num_layers,
        enc_num_heads,
        speech_max_len,
        enc_dropout,
        dec_num_layers,
        dec_num_heads,
        d_ff,
        dec_dropout,
        target_vocab_size,
        trans_max_len,
    ):

        super(Transformer, self).__init__()

        self.embedding = SpeechEmbedding(
            input_dim, d_model, time_stride, feature_stride, embed_dropout
        )
        speech_max_len = int(
            np.ceil(speech_max_len / self.embedding.time_downsampling_factor)
        )

        self.encoder = Encoder(
            enc_num_layers,
            d_model,
            enc_num_heads,
            d_ff,
            speech_max_len,
            target_vocab_size,
            enc_dropout,
        )

        self.decoder = Decoder(
            dec_num_layers,
            d_model,
            dec_num_heads,
            d_ff,
            dec_dropout,
            trans_max_len,
            target_vocab_size,
        )

    def forward(
        self,
        padded_input,
        input_lengths,
        padded_target,
        target_lengths,
        mode: Literal["full", "dec_cond_lm", "dec_lm"] = "full",
    ):
        """DO NOT MODIFY"""
        if mode == "full":  # Full transformer training
            encoder_output, encoder_lengths = self.embedding(
                padded_input, input_lengths, use_blstm=False
            )
            encoder_output, encoder_lengths, ctc_out = self.encoder(
                encoder_output, encoder_lengths
            )
        if mode == "dec_cond_lm":  # Training Decoder as a conditional LM
            encoder_output, encoder_lengths = self.embedding(
                padded_input, input_lengths, use_blstm=True
            )
            ctc_out = None
        if mode == "dec_lm":  # Training Decoder as an LM
            encoder_output, encoder_lengths, ctc_out = None, None, None

        # passing Encoder output through Decoder
        output, attention_weights = self.decoder(
            padded_target, target_lengths, encoder_output, encoder_lengths
        )
        return output, attention_weights, ctc_out

    def recognize(
        self,
        inp,
        inp_len,
        tokenizer,
        mode: Literal["full", "dec_cond_lm", "dec_lm"],
        strategy: str = "greedy",
    ):
        """sequence-to-sequence greedy search -- decoding one utterance at a time"""
        """DO NOT MODIFY"""
        if mode == "full":
            encoder_output, encoder_lengths = self.embedding(
                inp, inp_len, use_blstm=False
            )
            encoder_output, encoder_lengths, ctc_out = self.encoder(
                encoder_output, encoder_lengths
            )

        if mode == "dec_cond_lm":
            (
                encoder_output,
                encoder_lengths,
            ) = self.embedding(inp, inp_len, use_blstm=True)
            ctc_out = None

        if mode == "dec_lm":
            encoder_output, encoder_lengths, ctc_out = None, None, None

        if strategy == "greedy":
            out = self.decoder.recognize_greedy_search(
                encoder_output, encoder_lengths, tokenizer=tokenizer
            )
        elif strategy == "beam":
            out = self.decoder.recognize_beam_search(
                encoder_output, encoder_lengths, tokenizer=tokenizer, beam_width=5
            )
        return out


model = Transformer(
    input_dim=x_pad.shape[-1],
    time_stride=config["time_stride"],
    feature_stride=config["feature_stride"],
    embed_dropout=config["embed_dropout"],
    d_model=config["d_model"],
    enc_num_layers=config["enc_num_layers"],
    enc_num_heads=config["enc_num_heads"],
    speech_max_len=MAX_SPEECH_LEN,
    enc_dropout=config["enc_dropout"],
    dec_num_layers=config["dec_num_layers"],
    dec_num_heads=config["dec_num_heads"],
    d_ff=config["d_ff"],
    dec_dropout=config["dec_dropout"],
    target_vocab_size=Tokenizer.VOCAB_SIZE,
    trans_max_len=MAX_TRANS_LEN,
)

summary(
    model.to(device),
    input_data=[
        x_pad.to(device),
        x_len.to(device),
        y_shifted_pad.to(device),
        y_len.to(device),
    ],
)

gc.collect()
torch.cuda.empty_cache()


def calculateMetrics(reference, hypothesis):
    # sentence-level edit distance
    dist = aF.edit_distance(reference, hypothesis)
    # split sentences into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # compute edit distance
    dist = aF.edit_distance(ref_words, hyp_words)
    # calculate WER
    wer = dist / len(ref_words)
    # convert sentences into character sequences
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    # compute edit distance
    dist = aF.edit_distance(ref_chars, hyp_chars)
    # calculate CER
    cer = dist / len(ref_chars)
    return dist, wer * 100, cer * 100


def calculateBatchMetrics(predictions, y, y_len, tokenizer):
    """
    Calculate levenshtein distance, WER, CER for a batch
    predictions (Tensor) : the model predictions
    y (Tensor) : the target transcript
    y_len (Tensor) : Length of the target transcript (non-padded positions)
    """
    batch_size, _ = predictions.shape
    dist, wer, cer = 0.0, 0.0, 0.0
    for batch_idx in range(batch_size):

        # trim predictons upto the EOS_TOKEN
        pad_indices = torch.where(predictions[batch_idx] == tokenizer.EOS_TOKEN)[0]
        lowest_pad_idx = (
            pad_indices.min().item()
            if pad_indices.numel() > 0
            else len(predictions[batch_idx])
        )
        pred_trimmed = predictions[batch_idx, :lowest_pad_idx]

        # trim target upto EOS_TOKEN
        y_trimmed = y[batch_idx, 0 : y_len[batch_idx] - 1]

        # decodes
        pred_string = tokenizer.decode(pred_trimmed)
        y_string = tokenizer.decode(y_trimmed)

        # calculate metrics and update
        curr_dist, curr_wer, curr_cer = calculateMetrics(y_string, pred_string)
        dist += curr_dist
        wer += curr_wer
        cer += curr_cer

    # average by batch sizr
    dist /= batch_size
    wer /= batch_size
    cer /= batch_size
    return dist, wer, cer, y_string, pred_string


def save_attention_plot(
    plot_path,
    attention_weights,
    epoch=0,
    mode: Literal["full", "dec_cond_lm", "dec_lm"] = "full",
):
    """
    Saves attention weights plot to a specified path.

    Args:
        plot_path (str): Directory path where the plot will be saved.
        attention_weights (Tensor): Attention weights to plot.
        epoch (int): Current training epoch (default is 0).
        mode (str): Mode of attention - 'full', 'dec_cond_lm', or 'dec_lm'.
    """
    if not isinstance(attention_weights, (np.ndarray, torch.Tensor)):
        raise ValueError("attention_weights must be a numpy array or torch Tensor")

    plt.clf()  # Clear the current figure
    sns.heatmap(attention_weights, cmap="viridis", cbar=True)  # Create heatmap
    plt.title(f"{mode} Attention Weights - Epoch {epoch}")
    plt.xlabel("Target Sequence")
    plt.ylabel("Source Sequence")

    # Save the plot with clearer filename distinction
    attention_type = "cross" if epoch < 100 else "self"
    epoch_label = epoch if epoch < 100 else epoch - 100
    plt.savefig(f"{plot_path}/{mode}_{attention_type}_attention_epoch{epoch_label}.png")


def save_model(model, optimizer, scheduler, metric, epoch, path):
    """
    Saves the model, optimizer, and scheduler states along with a metric to a specified path.

    Args:
        model (nn.Module): Model to be saved.
        optimizer (Optimizer): Optimizer state to save.
        scheduler (Scheduler or None): Scheduler state to save.
        metric (tuple): Metric tuple (name, value) to be saved.
        epoch (int): Current epoch number.
        path (str): File path for saving.
    """
    # Ensure metric is provided as a tuple with correct structure
    if not (isinstance(metric, tuple) and len(metric) == 2):
        raise ValueError("metric must be a tuple in the form (name, value)")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else {},
            metric[0]: metric[1],  # Unpacks the metric name and value
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    checkpoint_path,
    model,
    embedding_load: bool,
    encoder_load: bool,
    decoder_load: bool,
    optimizer=None,
    scheduler=None,
):
    """
    Loads weights from a checkpoint into the model and optionally returns updated model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (Transformer): Transformer model to load weights into.
        embedding_load (bool): Load embedding weights if True.
        encoder_load (bool): Load encoder weights if True.
        decoder_load (bool): Load decoder weights if True.
        optimizer (Optimizer, optional): Optimizer to load state into (if provided).
        scheduler (Scheduler, optional): Scheduler to load state into (if provided).

    Returns:
        model (Transformer): Model with loaded weights.
        optimizer (Optimizer or None): Optimizer with loaded state if provided.
        scheduler (Scheduler or None): Scheduler with loaded state if provided.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()

    # Define the components to be loaded
    load_map = {
        "embedding": embedding_load,
        "encoder": encoder_load,
        "decoder": decoder_load,
    }

    # Filter and load the specified components
    for key, should_load in load_map.items():
        if should_load:
            component_state_dict = {
                k: v
                for k, v in checkpoint["model_state_dict"].items()
                if k.startswith(key)
            }
            if component_state_dict:
                model_state_dict.update(component_state_dict)
            else:
                print(f"Warning: No weights found for {key} in checkpoint.")

    # Load the updated state_dict into the model
    model.load_state_dict(model_state_dict, strict=False)
    loaded_components = ", ".join([k.capitalize() for k, v in load_map.items() if v])
    print(f"Loaded components: {loaded_components}")

    # Load optimizer and scheduler states if available and provided
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler


def train_step(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    ctc_loss: nn.CTCLoss,
    ctc_weight: float,
    optimizer,
    scheduler,
    scaler,
    device: str,
    train_loader: DataLoader,
    tokenizer: Any,
    mode: Literal["full", "dec_cond_lm", "dec_lm"],
) -> Tuple[float, float, torch.Tensor]:
    """
    Trains a model for one epoch based on the specified training mode.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.CrossEntropyLoss): The loss function for cross-entropy.
        ctc_loss (nn.CTCLoss): The loss function for CTC.
        ctc_weight (float): Weight of the CTC loss in the total loss calculation.
        optimizer (Optimizer): The optimizer to update model parameters.
        scheduler (_LRScheduler): The learning rate scheduler.
        scaler (GradScaler): For mixed-precision training.
        device (str): The device to run training on, e.g., 'cuda' or 'cpu'.
        train_loader (DataLoader): The training data loader.
        tokenizer (Any): Tokenizer with PAD_TOKEN attribute.
        mode (Literal): Specifies the training objective.

    Returns:
        Tuple[float, float, torch.Tensor]: The average training loss, perplexity, and attention weights.
    """
    model.train()
    batch_bar = tqdm(
        total=len(train_loader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc=f"[Train mode: {mode}]",
    )

    running_loss = 0.0
    running_perplexity = 0.0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Separate inputs and targets based on the mode
        if mode != "dec_lm":
            inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths = (
                batch
            )
            inputs = inputs.to(device)
        else:
            inputs, inputs_lengths = None, None
            targets_shifted, targets_golden, targets_lengths = batch

        targets_shifted = targets_shifted.to(device)
        targets_golden = targets_golden.to(device)

        # Forward pass with mixed-precision
        with torch.autocast(device_type=device, dtype=torch.float16):
            raw_predictions, attention_weights, ctc_out = model(
                inputs, inputs_lengths, targets_shifted, targets_lengths, mode=mode
            )
            padding_mask = torch.logical_not(
                torch.eq(targets_shifted, tokenizer.PAD_TOKEN)
            )

            # Calculate cross-entropy loss
            ce_loss = (
                criterion(raw_predictions.transpose(1, 2), targets_golden)
                * padding_mask
            )
            loss = ce_loss.sum() / padding_mask.sum()

            # Optionally optimize a weighted sum of ce and ctc_loss from the encoder outputs
            # Only available during full transformer training, a ctc_loss must be passed in
            if mode == "full" and ctc_loss and ctc_out is not None:
                inputs_lengths = torch.ceil(
                    inputs_lengths.float() / model.embedding.time_downsampling_factor
                ).int()
                inputs_lengths = inputs_lengths.clamp(max=ctc_out.size(0))
                loss += ctc_weight * ctc_loss(
                    ctc_out, targets_golden, inputs_lengths, targets_lengths
                )

        # Backward pass and optimization with mixed-precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss and perplexity for monitoring
        running_loss += float(loss.item())
        perplexity = torch.exp(loss)
        running_perplexity += perplexity.item()

        # Update the progress bar
        batch_bar.set_postfix(
            loss=f"{running_loss / (i + 1):.4f}",
            perplexity=f"{running_perplexity / (i + 1):.4f}",
        )
        batch_bar.update()

        # Clean up to save memory
        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()

    # Compute average loss and perplexity
    avg_loss = running_loss / len(train_loader)
    avg_perplexity = running_perplexity / len(train_loader)
    batch_bar.close()

    return avg_loss, avg_perplexity, attention_weights


def validate_step(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: str,
    mode: Literal["full", "dec_cond_lm", "dec_lm"],
    threshold: int = 5,
) -> Tuple[float, Dict[int, Dict[str, str]], float, float]:
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): Validation data loader.
        tokenizer (Any): Tokenizer with a method to handle special tokens.
        device (str): The device to run validation on, e.g., 'cuda' or 'cpu'.
        mode (Literal): Specifies the validation objective.
        threshold (int, optional): Max number of batches to validate on (for early stopping).

    Returns:
        Tuple[float, Dict[int, Dict[str, str]], float, float]: The average distance, JSON output with inputs/outputs,
                                                               average WER, and average CER.
    """
    model.eval()
    batch_bar = tqdm(
        total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc="Val"
    )

    running_distance = 0.0
    running_wer = 0.0
    running_cer = 0.0
    json_output = {}

    with torch.inference_mode():
        for i, batch in enumerate(val_loader):
            # Separate inputs and targets based on the mode
            if mode != "dec_lm":
                (
                    inputs,
                    targets_shifted,
                    targets_golden,
                    inputs_lengths,
                    targets_lengths,
                ) = batch
                inputs = inputs.to(device)
            else:
                inputs, inputs_lengths = None, None
                _, targets_shifted, targets_golden, _, targets_lengths = batch

            if targets_shifted is not None:
                targets_shifted = targets_shifted.to(device)
            if targets_golden is not None:
                targets_golden = targets_golden.to(device)

            # Perform recognition and calculate metrics
            greedy_predictions = model.recognize(
                inputs, inputs_lengths, tokenizer=tokenizer, mode=mode
            )
            dist, wer, cer, y_string, pred_string = calculateBatchMetrics(
                greedy_predictions, targets_golden, targets_lengths, tokenizer
            )

            # Accumulate metrics
            running_distance += dist
            running_wer += wer
            running_cer += cer
            json_output[i] = {"Input": y_string, "Output": pred_string}

            # Update progress bar
            batch_bar.set_postfix(
                running_distance=f"{running_distance / (i + 1):.4f}",
                WER=f"{running_wer / (i + 1):.4f}",
                CER=f"{running_cer / (i + 1):.4f}",
            )
            batch_bar.update()

            # Early stopping for thresholded validation
            if threshold and i == threshold:
                break

            del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
            torch.cuda.empty_cache()

    # Compute averages
    num_batches = threshold + 1 if threshold else len(val_loader)
    avg_distance = running_distance / num_batches
    avg_wer = running_wer / num_batches
    avg_cer = running_cer / num_batches
    batch_bar.close()

    return avg_distance, json_output, avg_wer, avg_cer


loss_func = nn.CrossEntropyLoss(ignore_index=Tokenizer.PAD_TOKEN)
ctc_loss_fn = None
if config["use_ctc"]:
    ctc_loss_fn = nn.CTCLoss(blank=Tokenizer.PAD_TOKEN)
scaler = torch.amp.GradScaler("cuda")


def get_optimizer():
    optimizer = None
    if config["optimizer"] == "SGD":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=1e-4,
            nesterov=config["nesterov"],
        )

    elif config["optimizer"] == "Adam":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.01
        )

    elif config["optimizer"] == "AdamW":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.01
        )
    return optimizer


optimizer = get_optimizer()
assert optimizer != None


def get_scheduler():
    scheduler = None
    if config["scheduler"] == "ReduceLR":
        # Feel Free to change any of the initializations you like to fit your needs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["factor"],
            patience=config["patience"],
            min_lr=5e-7,
            threshold=1e-1,
        )

    elif config["scheduler"] == "CosineAnnealing":
        # Feel Free to change any of the initializations you like to fit your needs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=1e-8
        )
    return scheduler


scheduler = get_scheduler()
assert scheduler != None

USE_WANDB = config["use_wandb"]
RESUME_LOGGING = False

# creating your WandB run
run_name = "{}_{}_Transformer_ENC-{}-{}_DEC-{}-{}_{}_{}_{}_{}_token_{}".format(
    config["Name"],
    config["feat_type"],
    config["enc_num_layers"],
    config["enc_num_heads"],
    config["dec_num_layers"],
    config["dec_num_heads"],
    config["d_model"],
    config["d_ff"],
    config["optimizer"],
    config["scheduler"],
    config["token_type"],
)

expt_root = os.path.join(os.getcwd(), run_name)
os.makedirs(expt_root, exist_ok=True)

if USE_WANDB:
    wandb.login(key="", relogin=True)  # TODO enter your key here

    if RESUME_LOGGING:
        run_id = ""
        run = wandb.init(
            id=run_id,  ### Insert specific run id here if you want to resume a previous run
            resume=True,  ### You need this to resume previous runs, but comment out reinit=True when using this
            project="HW4P2-Fall",  ### Project should be created in your wandb account
        )

    else:
        run = wandb.init(
            name=run_name,  ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit=True,  ### Allows reinitalizing runs when you re-run this cell
            project="HW4P2-Fall",  ### Project should be created in your wandb account
            config=config,  ### Wandb Config for your run
        )

        ### Save your model architecture as a string with str(model)
        model_arch = str(model)
        ### Save it in a txt file
        model_path = os.path.join(expt_root, "model_arch.txt")
        arch_file = open(model_path, "w")
        file_write = arch_file.write(model_arch)
        arch_file.close()

        ### Log it in your wandb run with wandb.sav


### Create a local directory with all the checkpoints
shutil.copy(
    os.path.join(os.getcwd(), "config.yaml"), os.path.join(expt_root, "config.yaml")
)
e = 0
best_loss = 10.0
best_perplexity = 10.0
best_dist = 60
RESUME_LOGGING = False
checkpoint_root = os.path.join(expt_root, "checkpoints")
text_root = os.path.join(expt_root, "out_text")
attn_img_root = os.path.join(expt_root, "attention_imgs")
os.makedirs(checkpoint_root, exist_ok=True)
os.makedirs(attn_img_root, exist_ok=True)
os.makedirs(text_root, exist_ok=True)
checkpoint_best_loss_model_filename = "checkpoint-best-loss-modelfull.pth"
checkpoint_last_epoch_filename = "checkpoint-epochfull-"
best_loss_model_path = os.path.join(
    checkpoint_root, checkpoint_best_loss_model_filename
)


if USE_WANDB:
    wandb.watch(model, log="all")

if RESUME_LOGGING:
    # change if you want to load best test model accordingly
    checkpoint = torch.load(
        wandb.restore(checkpoint_best_loss_model_filename, run_path="" + run_id).name
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    e = checkpoint["epoch"]

    print("Resuming from epoch {}".format(e + 1))
    print("Epochs left: ", config["epochs"] - e)
    print("Optimizer: \n", optimizer)

torch.cuda.empty_cache()
gc.collect()


print("Warming up entire model")

gc.collect()
torch.cuda.empty_cache()

# set your epochs for this approach
epochs = 4
for epoch in range(e, epochs):

    print("\nEpoch {}/{}".format(epoch + 1, epochs))

    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_step(
        model,
        criterion=loss_func,
        ctc_loss=None,
        ctc_weight=0.0,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_loader=train_loader,
        tokenizer=Tokenizer,
        mode="full",
    )

    print(
        "\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, epochs, train_loss, train_perplexity, curr_lr
        )
    )

    levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode="dec_cond_lm",
        threshold=5,
    )

    fpath = os.path.join(text_root, f"dec_cond_lm_{epoch+1}_out.json")
    with open(fpath, "w") as f:
        json.dump(json_out, f, indent=4)

    print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    print("WER                  : {:.04f}".format(wer))
    print("CER                  : {:.04f}".format(cer))

    attention_keys = list(attention_weights.keys())
    attention_weights_decoder_self = (
        attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    )
    attention_weights_decoder_cross = (
        attention_weights[attention_keys[-1]][0].cpu().detach().numpy()
    )

    if USE_WANDB:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "learning_rate": curr_lr,
                "lev_dist": levenshtein_distance,
                "WER": wer,
                "CER": cer,
            }
        )

    save_attention_plot(
        str(attn_img_root), attention_weights_decoder_cross, epoch, mode="dec_cond_lm"
    )
    save_attention_plot(
        str(attn_img_root),
        attention_weights_decoder_self,
        epoch + 100,
        mode="dec_cond_lm",
    )
    if config["scheduler"] == "ReduceLR":
        scheduler.step(levenshtein_distance)
    else:
        scheduler.step()

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    epoch_model_path = os.path.join(
        checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + ".pth")
    )
    save_model(model, optimizer, scheduler, ("CER", cer), epoch, epoch_model_path)

    if best_dist >= levenshtein_distance:
        best_loss = train_loss
        best_dist = levenshtein_distance
        save_model(
            model, optimizer, scheduler, ("CER", cer), epoch, best_loss_model_path
        )
        print("Saved best CER model")

gc.collect()
torch.cuda.empty_cache()

optimizer = get_optimizer()
scheduler = get_scheduler()

print("Pretraining Decoder")

# set your epochs for this approach
epochs = 24
for epoch in range(e, epochs):

    print("\nEpoch {}/{}".format(epoch + 1, epochs))

    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_step(
        model,
        criterion=loss_func,
        ctc_loss=None,
        ctc_weight=0.0,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_loader=train_loader,
        tokenizer=Tokenizer,
        mode="dec_cond_lm",
    )

    print(
        "\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, epochs, train_loss, train_perplexity, curr_lr
        )
    )

    levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode="dec_cond_lm",
        threshold=5,
    )

    fpath = os.path.join(text_root, f"dec_cond_lm_{epoch+1}_out.json")
    with open(fpath, "w") as f:
        json.dump(json_out, f, indent=4)

    print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    print("WER                  : {:.04f}".format(wer))
    print("CER                  : {:.04f}".format(cer))

    attention_keys = list(attention_weights.keys())
    attention_weights_decoder_self = (
        attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    )
    attention_weights_decoder_cross = (
        attention_weights[attention_keys[-1]][0].cpu().detach().numpy()
    )

    if USE_WANDB:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "learning_rate": curr_lr,
                "lev_dist": levenshtein_distance,
                "WER": wer,
                "CER": cer,
            }
        )

    save_attention_plot(
        str(attn_img_root), attention_weights_decoder_cross, epoch, mode="dec_cond_lm"
    )
    save_attention_plot(
        str(attn_img_root),
        attention_weights_decoder_self,
        epoch + 100,
        mode="dec_cond_lm",
    )
    if config["scheduler"] == "ReduceLR":
        scheduler.step(levenshtein_distance)
    else:
        scheduler.step()

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    epoch_model_path = os.path.join(
        checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + ".pth")
    )
    save_model(model, optimizer, scheduler, ("CER", cer), epoch, epoch_model_path)

    if best_dist >= levenshtein_distance:
        best_loss = train_loss
        best_dist = levenshtein_distance
        save_model(
            model, optimizer, scheduler, ("CER", cer), epoch, best_loss_model_path
        )
        print("Saved best CER model")

# freeze_embeddings
for name, param in model.named_parameters():
    if name.startswith("embedding"):
        param.requires_grad = False


# freeze decoder
for name, param in model.named_parameters():
    if name.startswith("decoder"):
        param.requires_grad = False

optimizer = get_optimizer()
scheduler = get_scheduler()

# Encoder warm up
print("Warming up encoder ")
gc.collect()
torch.cuda.empty_cache()


epochs = 4
for epoch in range(epochs):

    print("\nEpoch {}/{}".format(epoch + 1, epochs))

    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_step(
        model,
        criterion=loss_func,
        ctc_loss=ctc_loss_fn,
        ctc_weight=config["ctc_weight"],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_loader=train_loader,
        tokenizer=Tokenizer,
        mode="full",
    )

    print(
        "\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, epochs, train_loss, train_perplexity, curr_lr
        )
    )

    levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode="full",
        threshold=5,
    )

    fpath = os.path.join(text_root, f"full_{epoch+1}_out.json")
    with open(fpath, "w") as f:
        json.dump(json_out, f, indent=4)

    print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    print("WER                  : {:.04f}".format(wer))
    print("CER                  : {:.04f}".format(cer))

    attention_keys = list(attention_weights.keys())
    attention_weights_decoder_self = (
        attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    )
    attention_weights_decoder_cross = (
        attention_weights[attention_keys[-1]][0].cpu().detach().numpy()
    )

    if USE_WANDB:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "learning_rate": curr_lr,
                "lev_dist": levenshtein_distance,
                "WER": wer,
                "CER": cer,
            }
        )

    save_attention_plot(
        str(attn_img_root), attention_weights_decoder_cross, epoch, mode="full"
    )
    save_attention_plot(
        str(attn_img_root), attention_weights_decoder_self, epoch + 100, mode="full"
    )
    if config["scheduler"] == "ReduceLR":
        scheduler.step(levenshtein_distance)
    else:
        scheduler.step()

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    epoch_model_path = os.path.join(
        checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + "-2" + ".pth")
    )
    save_model(model, optimizer, scheduler, ("CER", cer), epoch, epoch_model_path)

    if best_dist >= levenshtein_distance:
        best_loss = train_loss
        best_dist = levenshtein_distance
        save_model(
            model, optimizer, scheduler, ("CER", cer), epoch, best_loss_model_path
        )
        print("Saved best distance model")

print("Unfreezing all layers")


# unfreeze_embeddings():
for name, param in model.named_parameters():
    if name.startswith("embedding"):
        param.requires_grad = True


# unfreeze_encoder():
for name, param in model.named_parameters():
    if name.startswith("encoder"):
        param.requires_grad = True


#  unfreeze_decoder():
for name, param in model.named_parameters():
    if name.startswith("decoder"):
        param.requires_grad = True

optimizer = get_optimizer()
scheduler = get_scheduler()

gc.collect()
torch.cuda.empty_cache()

epochs = config["epochs"]
for epoch in range(e, epochs):
    print("\nEpoch {}/{}".format(epoch + 1, epochs))
    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_step(
        model,
        criterion=loss_func,
        ctc_loss=ctc_loss_fn,
        ctc_weight=config["ctc_weight"],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_loader=train_loader,
        tokenizer=Tokenizer,
        mode="full",
    )
    print(
        "\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, epochs, train_loss, train_perplexity, curr_lr
        )
    )

    levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode="full",
        threshold=5,
    )

    fpath = os.path.join(text_root, f"full_{epoch+1}_out.json")
    with open(fpath, "w") as f:
        json.dump(json_out, f, indent=4)

    print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    print("WER                  : {:.04f}".format(wer))
    print("CER                  : {:.04f}".format(cer))

    attention_keys = list(attention_weights.keys())
    attention_weights_decoder_self = (
        attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    )
    attention_weights_decoder_cross = (
        attention_weights[attention_keys[-1]][0].cpu().detach().numpy()
    )

    if USE_WANDB:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "learning_rate": curr_lr,
                "lev_dist": levenshtein_distance,
                "WER": wer,
                "CER": cer,
            }
        )

    save_attention_plot(
        str(attn_img_root), attention_weights_decoder_cross, epoch, mode="full"
    )
    save_attention_plot(
        str(attn_img_root), attention_weights_decoder_self, epoch + 100, mode="full"
    )
    if config["scheduler"] == "ReduceLR":
        scheduler.step(levenshtein_distance)
    else:
        scheduler.step()

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    epoch_model_path = os.path.join(
        checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + "-2" + ".pth")
    )
    save_model(model, optimizer, scheduler, ("CER", cer), epoch, epoch_model_path)

    if best_dist >= levenshtein_distance:
        best_loss = train_loss
        best_dist = levenshtein_distance
        save_model(
            model, optimizer, scheduler, ("CER", cer), epoch, best_loss_model_path
        )
        print("Saved best distance model")

### Finish your wandb run
if USE_WANDB:
    run.finish()
# #### -------------

print("Loading best model")

model, _, _ = load_checkpoint(
    f"/home/yash/{run_name}/checkpoints/checkpoint-best-loss-modelfull.pth",
    model,
    True,
    True,
    True,
)

levenshtein_distance, json_out, wer, cer = validate_step(
    model,
    val_loader=val_loader,
    tokenizer=Tokenizer,
    device=device,
    mode="full",
    threshold=None,
)

print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
print("WER                  : {:.04f}".format(wer))
print("CER                  : {:.04f}".format(cer))


fpath = os.path.join(os.getcwd(), f"final_out_{run_name}.json")
with open(fpath, "w") as f:
    json.dump(json_out, f, indent=4)


def test_step(model, test_loader, tokenizer, device):
    model.eval()
    # progress bar
    batch_bar = tqdm(
        total=len(test_loader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Test",
        ncols=5,
    )

    predictions = []

    ## Iterate through batches
    for i, batch in enumerate(test_loader):

        inputs, _, _, inputs_lengths, _ = batch
        inputs = inputs.to(device)

        with torch.no_grad():
            greedy_predictions = model.recognize(
                inputs,
                inputs_lengths,
                tokenizer=tokenizer,
                mode="full",
                strategy="greedy",
            )

        # @NOTE: modify the print_example to print more or less validation examples
        batch_size, _ = greedy_predictions.shape
        batch_pred = []

        ## TODO decode each sequence in the batch
        for batch_idx in range(batch_size):
            pred_sequence = greedy_predictions[batch_idx].tolist()

            if tokenizer.EOS_TOKEN in pred_sequence:
                eos_index = pred_sequence.index(tokenizer.EOS_TOKEN)
                pred_sequence = pred_sequence[:eos_index]

            pred_string = tokenizer.decode(pred_sequence)
            batch_pred.append(pred_string)

        predictions.extend(batch_pred)

        batch_bar.update()

        del inputs, inputs_lengths
        torch.cuda.empty_cache()

    return predictions


predictions = test_step(
    model,
    test_loader=test_loader,
    tokenizer=Tokenizer,
    device=device,
)

import csv

# Specify the CSV file path
csv_file_path = "submission.csv"

# Write the list to the CSV with index as the first column
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Index", "Labels"])
    # Write list items with index
    for idx, item in enumerate(predictions):
        writer.writerow([idx, item])

print(f"CSV file saved to {csv_file_path}")
