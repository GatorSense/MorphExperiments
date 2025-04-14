# Morphological Neural Networks (MNN)

This is the repository for the current experiments on MNNs in the Machine Learning and Sensing Lab. These experiments are a continuation of the work done in Xu (2023).

## Instructions To Run

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

The following metrics are logged by train.py:
  - Training and testing confusion matrices
  - Histograms of feature map values throughout training per class
  - True vs. prediction label heatmap on all test classes
  - Hit and miss filters over training

All of the above are viewed on [Comet.ml](https://www.comet.ml). To use this functionality, store the API key in .env file at the root directory with the following format:

```ini
COMET_API_KEY=your_api_key_here
```

## Project Structure

```bash
MorphExperiments/
├─ models/
│  ├─ MNN.py             # Implementation of learnable Hit-Miss transformation
│  ├─ models.py          # Model classes implementation (MNN, CNN, MCNN)
├─ utils/
│  ├─ custom_dataset.py  # Custom datasets for binary classifiers
│  ├─ logger.py          # Function to log model weights thorugh training
│  ├─ plot.py            # Plots feature map histograms and filters through training
├─ .gitignore
├─ train.py              # Script to run for training loop
└─ README.md
```

## References
Xu, Weihuang (2023). Deep Morph-Convolutional Neural Network: Combining Morphological Transform and Convolution in Deep Neural Networks. [Doctoral dissertation, University of Florida]. UF Digital Collections. https://ufdc.ufl.edu/UFE0059487/00001/pd
