# CIFAR10 Classification with Dilated CNN

A custom CNN implementation for CIFAR10 image classification using dilated convolutions and depthwise separable convolutions.

## Requirements

- Python 3.8-3.10 (Not yet compatible with Python 3.11+ due to NumPy 2.0 constraints)
- CUDA-capable GPU recommended (but not required)

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```


2. Install dependencies:

```bash
pip install -r requirements.txt
```


## Usage

To train the model:

```bash
python train.py
```


## Model Architecture (CIFAR10Model)

The model uses a progressive dilated convolution approach with the following structure:

### Network Blocks
1. **Initial Block** (RF: 5 -> 9)
   - Conv(3→16, dilation=1) + BN + ReLU
   - Conv(16→24, dilation=2) + BN + ReLU

2. **Middle Block** (RF: 11 -> 19)
   - DepthwiseSeparable(24→32) + BN + ReLU
   - Conv(32→48, dilation=4) + BN + ReLU

3. **Feature Block** (RF: 35 -> 67)
   - Conv(48→64, dilation=8) + BN + ReLU
   - Conv(64→96, dilation=16) + BN + ReLU

4. **Final Block** (RF: 131)
   - Conv(96→128, dilation=32) + BN + ReLU

5. **Classification Head**
   - Global Average Pooling
   - Flatten
   - Linear(128→10)

### Key Features
- Maintains spatial dimensions (32x32) throughout the network
- Progressive dilation rates: 1 → 2 → 4 → 8 → 16 → 32
- Receptive field growth: 5 → 9 → 11 → 19 → 35 → 67 → 131
- Uses depthwise separable convolutions for efficiency
- Total parameters: < 200k

## Data Augmentation

Using Albumentations library with the following transformations:

### Training Transforms
1. **Horizontal Flip**
   - Probability: 50%

2. **ShiftScaleRotate**
   - Shift limit: ±10%
   - Scale limit: ±10%
   - Rotate limit: ±15 degrees
   - Probability: 50%

3. **CoarseDropout**
   - Holes: 1
   - Height/Width: 16px
   - Fill value: Dataset mean (0.4914, 0.4822, 0.4465)
   - Probability: 50%

4. **Normalization**
   - Mean: (0.4914, 0.4822, 0.4465)
   - Std: (0.2470, 0.2435, 0.2616)

### Test Transforms
- Only normalization is applied

## Implementation Details

### Custom Components

1. **DepthwiseConv**
   - Performs spatial convolution independently for each channel
   - Supports dilation and padding
   - Groups = input channels

2. **PointwiseConv**
   - 1x1 convolution for channel-wise mixing
   - Efficient parameter usage

3. **ConvBlock**
   - Modular block combining convolution, batch norm, and activation
   - Supports both regular and depthwise separable convolutions
   - Configurable dilation rate

### Training Setup
- Optimizer: Adam
- Learning Rate: 0.001
- Weight Decay: 1e-4
- Batch Size: 128
- Target Accuracy: 85%+
- Maximum Epochs: 40

### Project Structure

- `models/network.py`: Custom network components
- `utils/data_loader.py`: Data loading and augmentation
- `train.py`: Training script
- `requirements.txt`: Dependencies
- `README.md`: Project documentation


## License
[MIT License](LICENSE)