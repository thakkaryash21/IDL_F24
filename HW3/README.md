# Assignment 3

## Part 1

Implemented the following:

- RNN Cells
- GRU Cells
- CTCLoss

## Part 2

Link to the [Kaggle Competition](https://www.kaggle.com/competitions/11-785-hw3p2-f24/leaderboard?search=veri)

Developed an automatic speech recognition system using recurrent neural networks, converting audio utterances directly to phoneme sequences. The project involved:

- Implementing a sequence-to-sequence architecture combining convolutional and bidirectional LSTM networks to process variable-length speech inputs
- Building an end-to-end pipeline for phoneme recognition using Connectionist Temporal Classification (CTC) loss
- Applying data augmentation techniques including frequency and time masking to improve model robustness
- Achieving strong performance in a competitive evaluation using Levenshtein distance metrics on a Kaggle leaderboard

Model Details:

- Encoder: ResNet34 + BiLSTM
- Decoder: Linear
- CTC Loss
- Frequency Masking and Time Masking
