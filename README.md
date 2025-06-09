# Maize Leaf Disease Classification Submission by AgMo(Ken Ganza)

## Description
This project implements a Convolutional Neural Network (CNN) to classify maize leaf images into four disease categories, leveraging a dataset processed and visualized in a Colab notebook. The solution includes a FastAPI backend with Swagger UI for deployment, offering a scalable prediction API.

## GitHub Repository
- [Repository Link](https://github.com/kennyg37/AgMo-bn)

## Dataset link
[Link to dataset](https://huggingface.co/datasets/aquib1011/maize-leaf-disease)

## How to Set Up the Environment and Project
1. Clone the repository: `git clone https://github.com/kennyg37/AgMo-bn.git`
2. Navigate to the project folder: `cd maize-leaf-classification`
3. Install dependencies: `pip install -r requirements.txt`
4. Mount Google Drive in Colab (if using): Add `from google.colab import drive; drive.mount('/content/drive')`
5. Run the FastAPI app: `python main.py` (ensure model at `/content/drive/MyDrive/maize_leaf_cnn_model.h5`)
6. Access Swagger UI at `http://localhost:8000/docs` for testing.

## Designs
- [Swagger UI Screenshot](assets/swagger.png) (showing API endpoint)
- [Confusion Matrix Visualization](assets/confusion.png)

## Deployment Plan
The project is deployed as a FastAPI API with Swagger UI for real-time maize leaf disease prediction. Future steps include containerizing with Docker, hosting on a cloud platform like AWS or Heroku, and integrating with a mobile/web frontend for broader accessibility.

## Video Demo
- [Video Demo Link](https://drive.google.com/drive/folders/1Hqso-m_3w-i6uBDclv5AkZuwnxCknawp?usp=sharing) 

## ML Track - Model Notebook Content

### Data Visualization and Data Engineering
- Visualizes class distribution with a bar chart, showing 900-1000 samples for classes 0 and 3, 400 for class 2, and 1000 for class 1, highlighting imbalance.
- Applies `ImageDataGenerator` for augmentation (rotation, shifts, zoom) and splits data into training (2680) and validation (670) sets.

### Model Architecture
- Features a CNN with three Conv2D layers (32, 64, 128 filters), MaxPooling2D, a Dense layer (128 units with ReLU), Dropout (0.5), and a softmax output layer.
- Compiled with Adam optimizer and categorical crossentropy loss, optimized for 128x128x3 input images.

### Initial Performance Metrics
- Achieves a test accuracy of 0.8878 and test loss of 0.2784.
- Classification report shows precision/recall: 0.78/0.88 (class 0), 0.94/0.94 (class 1), 0.75/0.55 (class 2), 1.00/1.00 (class 3); weighted F1-score: 0.88.

### Deployment Option
- Provides a FastAPI backend with Swagger UI for API testing, allowing image uploads to predict maize leaf disease classes.