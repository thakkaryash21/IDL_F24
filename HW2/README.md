# Assignment 2

## Part 1

Implemented the following:

- Convolutional Layers
- Max Pooling Layers
- Down Sampling Layers
- Up Sampling Layers

## Part 2

Link to the [Kaggle Competition](https://www.kaggle.com/competitions/11785-hw-2-p-2-face-verification-fall-2024-slack/leaderboard?search=Its)

Designed and implemented a facial recognition system focusing on face classification and verification using Convolutional Neural Networks.

- Building custom CNN architectures based on ResNet to extract discriminative facial features from images
- Developing a complete verification pipeline that could determine if two face images belonged to the same person
- Applying advanced data augmentation techniques (random flips, rotations, color jittering) to improve model robustness
- Optimizing model performance using SGD with Cosine Annealing Warm Restarts scheduling
- Achieving high accuracy on both closed-set identification and challenging open-set verification tasks
- Participating in a competitive Kaggle leaderboard evaluation measuring Equal Error Rate (EER) and got a score of 0.09

Model Details:

- Resnet18
- Optimizer: SGD
- Scheduler: Cosine Annealing Warm Restarts
- Data Augmentation: Random Horizontal Flip, Random Rotation, Color Jitter, Random Affine, Random Erasing
