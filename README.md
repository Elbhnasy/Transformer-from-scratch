# Transformer Implementation from Scratch

This repository contains a clean, from-scratch implementation of the Transformer architecture, as introduced in the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. The goal is to build a deep understanding of the model by implementing each component manually without relying on high-level deep learning libraries.

## Features
- Encoder and decoder stacks
- Scaled dot-product attention and multi-head attention
- Positional encoding
- Masking mechanisms
- Fully connected layers with layer normalization and dropout
- PyTorch-based, but avoids using `nn.Transformer` or similar high-level modules
- Clear, modular code structure for better understanding
- Detailed comments explaining the mathematical operations

## Installation & Usage

### Prerequisites
- Python 3.12+
- PyTorch 2.6+


### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Transformer-from-scratch.git
cd Transformer-from-scratch
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Model
```bash
python train.py --data_path path/to/data --epochs 10
```

## Project Structure
```
Transformer-from-scratch/
├── model/
│   ├── __init__.py
│   ├── attention.py        # Multi-head attention implementation
│   ├── encoder.py          # Transformer encoder
│   ├── decoder.py          # Transformer decoder
│   ├── positional.py       # Positional encoding
│   ├── feed_forward.py     # Feed-forward networks
│   └── transformer.py      # Full transformer model
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   └── masking.py          # Padding and lookahead mask utilities
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Project dependencies
└── README.md
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT
