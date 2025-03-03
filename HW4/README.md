# Assignment 3

## Part 1

Implemented the following:

- Self Attention
- Next Token Prediction
- Generation of Sequence

## Part 2

Link to the [Kaggle Competition](https://www.kaggle.com/competitions/11-785-hw3p2-f24/leaderboard?search=veri)

Developed an end-to-end speech recognition system using Transformer architecture with multi-head attention mechanisms to convert speech recordings into accurate text transcriptions. The project involved:

- Implemented an attention-based end-to-end speech recognition system using the Transformer architecture to convert speech recordings into text transcriptions
- Designed and trained a model to process audio feature vectors from the LibriSpeech dataset, achieving competitive character error rates
- Built an encoder-decoder architecture featuring multi-head attention mechanisms, positional encoding, and cross-attention components
- Optimized hyperparameters and architectural design choices to balance computational efficiency and model performance

Model Details:

- Speech Embedding: CNN Subsampling + BiLSTM + Downsample
- Transformer Encoder: Multi-Head Self Attention + Feed Forward
- Transformer Decoder: Multi-Head Self Attention + Feed Forward + Cross Attention
