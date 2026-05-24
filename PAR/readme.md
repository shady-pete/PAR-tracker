# Model training
There are two training scripts present, train_1_attempt.py and train_2_attempt.py, differences and perfomance metrics are discussed in the report.
## Project Structure
Before running the script, be sure that the training and validation folder are in the same folder. Below the default project structure organization.
```
project_root/
│
├── Dataset/
│   ├── training/
│   │   ├── image/
│   │   │   └── ... (training images)
│   │   └── training_set.txt
│   │
│   └── validation/
│       ├── image/
│       │   └── ... (validation images)
│       └── validation_set.txt
│
├── PAR/
│   ├── train.py
│   ├── custom_dataset.py
│   └── custom_sampler.py
│
└── a1_models/
    └── model.pth
```

## Usage

```bash
python train_x_attempt.py [arguments]
```

## Arguments

| Argument | Flag | Type | Default | Description |
|----------|------|------|---------|-------------|
| Image Directory | `--image` | str | `../Dataset/training/image` | Directory containing training images |
| Labels File | `--labels` | str | `../Dataset/training/training_set.txt` | Path to training labels file |
| Epochs | `--epochs` | int | 100 | Number of training epochs |
| Learning Rate | `--learning_rate`, `-lr` | float | 0.001 | Model learning rate |
| Batch Size | `--batch_size`, `-bs` | int | 32 | Training batch size |
| Load Model | `--load_model` | str | `./a1_models/model.pth` | Path to load/save model weights |
| Max Samples | `--max_samples`, `-ms` | int | -1 | Maximum number of training samples (-1 for all samples) |



## Example

Train the model using custom parameters:

```bash
python train.py --image /path/to/images \
                --labels /path/to/labels.txt \
                --epochs 200 \
                -lr 0.0001 \
                -bs 64 \
                -ms 1000
```

Note: The validation dataset path is automatically derived by replacing "training" with "validation" in the provided image and labels paths.
Note: An already trained model is present in ../tracker/net , the file name is par_model.pth