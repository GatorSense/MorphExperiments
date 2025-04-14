# Morphological Neural Networks (MNN)

This is the repository for the current experiments on MNNs in the Machine Learning and Sensing Lab. These experiments are a continuation of the work done in Xu (2023).

## Instructions To Run

Clone this repository:
```bash
git clone https://github.com/GatorSense/MorphExperiments
cd MorphExperiments
```

Use the train.py file to train a binary classifier on hit-miss filters on MNIST images of threes and not threes (fours and eights). The test accuracy will include all 10 classes found in the MNIST data, showcasing its ability to reject classes, even if not included in the training data.

For example:

```bash
python train.py --epochs 100 --lr 0.1 --use-comet
```

These are all arguments used in our script:

```python
parser = argparse.ArgumentParser(description='PyTorch MNIST with MNNV2')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', type=str, default='morph', metavar='N',
                    help='type of layer to use (default: morph, could use conv or MCNN)')
parser.add_argument('--use-comet', action='store_true', default=False,
                    help='uses comet.ml to log training metrics and graphics')
```

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
