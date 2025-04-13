# Morphological Neural Networks

This is the repository for the current experiments on MNNs in the Machine Learning and Sensing Lab.

## Instructions to run

Clone this repository:
```bash
git clone https://github.com/GatorSense/MorphExperiments
cd MorphExperiments
```

Use the train.py file to train a binary classifier on hit-miss filters on MNIST images of threes and not threes (fours and eights). The test accuracy will include all 10 classes found in the MNIST data, showcasing its ability to reject classes, even if not included in the training data.

```bash
python train.py [args]
```

Refer to the help sections of the argument parser in train.py for more details.

## Metrics

The following metrics are logged throughout training:
  - Training and testing confusion matrices
  - Histograms of feature map values throughout training per class
  - Heatmap on all test classes
  - Hit and miss filters over training

All of the above are viewed on [Comet.ml](https://www.comet.ml), and the user's API key must be put in a .env file.
