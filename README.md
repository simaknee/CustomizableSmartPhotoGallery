# CustomizableSmartPhotoGallery
## Overview
CustomizableSmartPhotoGallery is an automatic image classifier for a smart photo gallery that uses a few-shot classification method based on a Siamese network to automatically classify images into user-defined categories. This repository focuses on training the model, converting it to ONNX, and performing inference tests. The application built using this model can be found in the [SmartGalleryApp](https://github.com/simaknee/CustomizableSmartPhotoGallery) repository.

## Development Environment
- Windows 10
- Python 3.10.16
- Pytorch 2.4.1 with CUDA 12.0
- ONNX Runtime 1.21.0

## Installation

Clone the repository and install the required libraries using the following commands:
```
git clone https://github.com/simaknee/CustomizableSmartPhotoGallery.git
cd CustomizableSmartPhotoGallery
pip install -r requirements.txt
```
If you're using a virtual environment, the `albumentations` library may cause an error. In that case, install it separately with the command below:
```
python -m pip install --user albumentations
```

## Dataset Preparation
The [mini-imagenet](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) dataset was used for training and validation. You can also create or use other datasets as long as they follow the same structure as mini-imagenet:
```
📦dataset_folder
 ┣ 📂{class_name1}
 ┃ ┣ 📜{image_file_name1}.JPEG
 ┃ ┣ 📜{image_file_name2}.JPEG
 ┃ ┣ 📜{image_file_name3}.JPEG
 ┃ ┣ ...
 ┣ 📂{class_name2}
 ┃ ┣ 📜{image_file_name1}.JPEG
 ┃ ┣ 📜{image_file_name2}.JPEG
 ┃ ┣ 📜{image_file_name3}.JPEG
 ...

```

## How to Run
### Training
Run the command below to train the model with the given training and validation datasets:
```
python train.py --train_path {YOUR_TRAIN_DATASET_PATH} --val_path {YOUR_VAL_DATASET_PATH} --epoch 500
```

### Evaluation
You can evaluate a pre-trained checkpoint file using the command below:
```
python eval.py --val_path {YOUR_VAL_DATASET_PATH} --checkpoint_file {YOUR_CHECKPOINT_FILE_PATH}
```

### ONNX Conversion
Convert the trained checkpoint file to ONNX format with the command below:
```
python export_onnx.py --checkpoint_path {YOUR_CHECKPOINT_FILE_PATH} --export_path {YOUR_PATH_TO_EXPORT}
```

## Training Results
The results below were obtained by training and validating the model on the mini-imagenet dataset.

![Image](https://github.com/user-attachments/assets/e11036d1-526e-409e-b63d-8ccbe18f3bbb)

The best performance was achieved at epoch 368, and the confusion matrix for 1000 validation samples is shown below:

|  | 	Actually Same | 	Actually Different |
| --- | --- | --- |
| <b>Predicted as Same</b> | 470 (True Positive) | 73 (False Positive) |
| <b>Predicted as Different</b> | 30 (False Negative) | 427 (True Negative) |

*(Based on the best threshold = 0.28)*

The accuracy was 89.7%, and the F1 Score was 0.901.

## Inference Test
You can check individual inference tests in the [inference.ipynb](inference.ipynb) file.


Since the paths and categories are set up for my environment, if you want to test it yourself, you need to define the dataset path, the trained model file path, and the category information in the file.

Below is a visualization of the inference test conducted on mini-imagenet.
![Image](https://github.com/user-attachments/assets/f61bcd3e-3a7d-433d-83b7-34534e8bd76c)
![Image](https://github.com/user-attachments/assets/5fa4264c-f977-40a7-a8fd-0095717cfa9f)
![Image](https://github.com/user-attachments/assets/2ca0f1fd-810c-44aa-9fe3-d910822cc109)

The following is an inference test where the model was used to concept of rhythm game result image classification.
![Image](https://github.com/user-attachments/assets/75a62c92-68e5-4697-9c15-b7bb182afb7d)
![Image](https://github.com/user-attachments/assets/9d797375-045e-4883-ba0d-9d8d31bee223)
![Image](https://github.com/user-attachments/assets/2236c57a-fe0e-4507-ac44-c926a8d9b916)