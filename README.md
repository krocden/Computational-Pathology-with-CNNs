

Computational Pathology with CNNs[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HsvvgckLQigXydq_PK6ACCfuw2YZhZ1i?usp=sharing)


This repository contains the research and development of machine learning models, particularly Convolutional Neural Networks (CNNs), applied to computational pathology and computer vision tasks.

## Project Overview

We aim to leverage CNNs to automate tissue classification in computational pathology and explore applications in computer vision, such as animal face classification. Our primary goal is to achieve high accuracy and address inherent challenges in the datasets.

## Datasets

1. **Colorectal Cancer Classification**: A reduced dataset containing 6K image patches split into three distinct classes.
2. **Prostate Cancer Classification**: Comprises 6k image patches from a larger dataset of 120k images, classifying tissue types.
3. **Animal Faces Classification**: Focuses on 6k images out of 16k, classifying animal types such as Cats, Dogs, and wildlife animals.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repo:
```bash
git clone https://github.com/krocden/COMP-432
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. run the `_init.py` the check the cuda condition to see if you are capable to use GPU acceleration
2. run the model with tweaked hyperparameters, once the model is trained, it will generate a file.
like `*modelname*_Seed=*seed*.pth` in the `Model/Saved` folder, this is the checkpoint of the model.
3. run the `ReportDriver.py` with desired seed, which you can find it in `Model/Saved/hyperparameter_log`,
then you can observe the output and the t-SNE report

## Challenges & Solutions

- **Data Variability**: Addressed through data preprocessing and augmentation techniques.
- **Imbalanced Datasets**: Utilized techniques like SMOTE and undersampling to balance the classes.
- **Transfer Learning**: Conducted experiments to fine-tune models across different datasets.
- **Computational Demands**: Recommended usage of GPU acceleration for training deep CNNs.

Certainly! Here's the updated "Team Members" table with the provided information:


## Team Members

| Name            | ID       | GitHub                              |
|-----------------|----------|-------------------------------------|
| Jonathan Haddad | 40111053 | [jonathangithubofficial](https://github.com/jonathangithubofficial) |
| Jainil Jaha     | 40067468 | [jjaha99](https://github.com/jjaha99)                   |
| Yixin Liu       | 40115632 | [Jiejue233](https://github.com/Jiejue233)                |
| Kevin Hong      | 40176625 | [krocden](https://github.com/krocden)                   |
| Yichen Huang     | 40167688 | [prprtracy](https://github.com/prprtracy)                |

---
## License

[MIT](https://choosealicense.com/licenses/mit/)
