import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from glob import glob
from skimage.io import imread
from os import listdir
import time
import copy
from tqdm import tqdm_notebook as tqdm
import streamlit as st
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
# from keras.models import model_from_json
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.models import Sequential
# from vit_keras import vit

# Global configurations
BATCH_SIZE = 32
NUM_CLASSES = 2
base_path = '/content/drive/MyDrive/Breast_cancer_project/Breast_cancer_patient/'
OUTPUT_PATH = ""
MODEL_PATH = "/content/drive/MyDrive/Breast_cancer_project/breast_data/"
LOSSES_PATH = "/content/drive/MyDrive/Breast_cancer_project/breast_data/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model initialization
model = torchvision.models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
model = model.to(device)
if device == "cpu":
    load_path = MODEL_PATH + ".pth"
else:
    load_path = MODEL_PATH + "_cuda.pth"

model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
model.eval()

# # Load ViT model (commented out as in original)
# vit_model = vit.vit_b32(
#     image_size=224,
#     activation='softmax',
#     pretrained=True,
#     include_top=False,
#     pretrained_top=False
# )

# # Load CNN model (commented out as in original)
# cnn_model = tf.keras.models.load_model('/content/drive/MyDrive/Breast_cancer_project/model.h5')

torch.manual_seed(0)
np.random.seed(0)

# Data utilities
def get_cancer_dataframe(patient_id, cancer_id):
    path = base_path + patient_id + "/" + cancer_id
    files = listdir(path)
    dataframe = pd.DataFrame(files, columns=["filename"])
    path_names = path + "/" + dataframe.filename.values
    dataframe = dataframe.filename.str.rsplit("_", n=4, expand=True)
    dataframe.loc[:, "target"] = int(cancer_id)
    dataframe.loc[:, "path"] = path_names
    dataframe = dataframe.drop([0, 1, 4], axis=1)
    dataframe = dataframe.rename({2: "x", 3: "y"}, axis=1)
    dataframe.loc[:, "x"] = dataframe.loc[:,"x"].str.replace("x", "", case=False).astype(int)
    dataframe.loc[:, "y"] = dataframe.loc[:,"y"].str.replace("y", "", case=False).astype(int)
    return dataframe

def get_patient_dataframe(patient_id):
    df_0 = get_cancer_dataframe(patient_id, "0")
    df_1 = get_cancer_dataframe(patient_id, "1")
    patient_df = pd.concat((df_0, df_1), ignore_index=True)
    return patient_df

def get_patient_dataframe1(patient_path):
    patient_id = patient_path.split('/')[-1]
    cancer_id = listdir(patient_path)
    lister = []
    for i in range(2):
        path = patient_path + "/" + cancer_id[i]
        files = listdir(path)
        dataframe = pd.DataFrame(files, columns=["filename"])
        path_names = path + "/" + dataframe.filename.values
        dataframe = dataframe.filename.str.rsplit("_", n=4, expand=True)
        dataframe.loc[:, "target"] = int(cancer_id[i])
        dataframe.loc[:, "path"] = path_names
        dataframe = dataframe.drop([0, 1, 4], axis=1)
        dataframe = dataframe.rename({2: "x", 3: "y"}, axis=1)
        dataframe.loc[:, "x"] = dataframe.loc[:,"x"].str.replace("x", "", case=False).astype(int)
        dataframe.loc[:, "y"] = dataframe.loc[:,"y"].str.replace("y", "", case=False).astype(int)
        dataframe.loc[:, "patient_id"] = str(patient_id)
        lister.append(dataframe)
    dataframe = pd.concat(lister, ignore_index=True)
    return dataframe

def extract_coords(df):
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = coord.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    coord.loc[:, "y"] = coord.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df

class BreastCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.states = df
        self.transform = transform
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        patient_id = self.states.patient_id.values[idx]
        x_coord = self.states.x.values[idx]
        y_coord = self.states.y.values[idx]
        image_path = self.states.path.values[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if "target" in self.states.columns.values:
            target = int(self.states.target.values[idx])
        else:
            target = None
            
        return {"image": image,
                "label": target,
                "patient_id": patient_id,
                "x": x_coord,
                "y": y_coord}

# Transformation and evaluation
def my_transform(key="train", plot=False):
    train_sequence = [transforms.Resize((50,50)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip()]
    val_sequence = [transforms.Resize((50,50))]
    if plot == False:
        train_sequence.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        val_sequence.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    data_transforms = {'train': transforms.Compose(train_sequence), 'val': transforms.Compose(val_sequence)}
    return data_transforms[key]

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image

def sigmoid(x):
    return 1./(1+np.exp(-x))

def f1_score(preds, targets):
    tp = (preds * targets).sum()
    fp = ((1 - targets) * preds).sum()
    fn = (targets * (1 - preds)).sum()
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    return f1_score

def evaluate_test_model(model, dataloader, predictions_df, key):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data["image"].to(device)
            labels = data["label"].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            proba = outputs.cpu().numpy().astype(float)
            predictions_df.loc[i*BATCH_SIZE:(i+1)*BATCH_SIZE-1, "proba"] = sigmoid(proba[:, 1])
            predictions_df.loc[i*BATCH_SIZE:(i+1)*BATCH_SIZE-1, "true"] = data["label"].numpy().astype(int)
            predictions_df.loc[i*BATCH_SIZE:(i+1)*BATCH_SIZE-1, "predicted"] = preds.cpu().numpy().astype(int)
            predictions_df.loc[i*BATCH_SIZE:(i+1)*BATCH_SIZE-1, "x"] = data["x"].numpy()
            predictions_df.loc[i*BATCH_SIZE:(i+1)*BATCH_SIZE-1, "y"] = data["y"].numpy()
            predictions_df.loc[i*BATCH_SIZE:(i+1)*BATCH_SIZE-1, "patient_id"] = data["patient_id"]
    predictions_df = predictions_df.dropna()
    return predictions_df

# Visualization functions
def visualise_breast_tissue(patient_id, pred_df=None):
    example_df = get_patient_dataframe(patient_id)
    max_point = [example_df.y.max()-1, example_df.x.max()-1]
    grid = 255*np.ones(shape = (max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    mask = 255*np.ones(shape = (max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    if pred_df is not None:
        patient_df = pred_df[pred_df.patient_id == patient_id].copy()
    mask_proba = np.zeros(shape = (max_point[0] + 50, max_point[1] + 50, 1)).astype(float)
    broken_patches = []
    for n in range(len(example_df)):
        try:
            image = imread(example_df.path.values[n])
            target = example_df.target.values[n]
            x_coord = int(example_df.x.values[n])
            y_coord = int(example_df.y.values[n])
            x_start = x_coord - 1
            y_start = y_coord - 1
            x_end = x_start + 50
            y_end = y_start + 50
            grid[y_start:y_end, x_start:x_end] = image
            if target == 1:
                mask[y_start:y_end, x_start:x_end, 0] = 250
                mask[y_start:y_end, x_start:x_end, 1] = 0
                mask[y_start:y_end, x_start:x_end, 2] = 0
            if pred_df is not None:
                proba = patient_df[(patient_df.x == x_coord) & (patient_df.y == y_coord)].proba
                mask_proba[y_start:y_end, x_start:x_end, 0] = proba
        except ValueError:
            broken_patches.append(example_df.path.values[n])
    return grid, mask, broken_patches, mask_proba

def classify_patch(patch):
    patch = patch.resize((224, 224))
    patch = np.array(patch) / 255.0
    patch = np.expand_dims(patch, axis=0)
    result = loaded_model.predict(patch)
    if result > 0.5:
        return 'Cancerous'
    else:
        return 'Non-cancerous'

def visualize(features):
    features = features.reshape((28, 28))
    features = np.repeat(np.repeat(features, 32, axis=0), 32, axis=1)
    shape = (7, 7, 768)
    if np.prod(shape) != features.size:
        raise ValueError(f"Cannot reshape {features.shape} into {shape}")
    features = features.reshape(shape)
    cam = softmax(features, axis=2)
    heatmap = np.sum(cam, axis=2)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = cv2.resize(heatmap, (224, 224))
    return heatmap

def visualize_tissue(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    features = vit_model.predict(image)
    if features is not None:
        st.write(features.shape[1])
    heatmap = visualize(features)
    st.image(heatmap, caption='Breast tissue heatmap')

def get_confusion_matrix(y_true, y_pred):
    transdict = {1: "cancer", 0: "no cancer"}
    y_t = np.array([transdict[x] for x in y_true])
    y_p = np.array([transdict[x] for x in y_pred])
    labels = ["no cancer", "cancer"]
    index_labels = ["actual no cancer", "actual cancer"]
    col_labels = ["predicted no cancer", "predicted cancer"]
    confusion = confusion_matrix(y_t, y_p, labels=labels)
    confusion_df = pd.DataFrame(confusion, index=index_labels, columns=col_labels)
    for n in range(2):
        confusion_df.iloc[n] = confusion_df.iloc[n] / confusion_df.sum(axis=1).iloc[n]
    return confusion_df

# Streamlit interface
def app_patient_visuals():
    st.header("Patient Tissue Visualization", help="Select a patient folder to visualize breast tissue and cancer predictions.")
    folder_path = '/content/drive/MyDrive/Breast_cancer_project/Breast_cancer_patient'
    patient_folders = os.listdir(folder_path)
    
    if not patient_folders:
        st.error("No patient folders found. Please check the data directory.")
        return
    
    patient_folder = st.selectbox('Select Patient', ['Select a patient...'] + patient_folders, help="Choose a patient to visualize their tissue data.")
    
    if patient_folder == 'Select a patient...':
        st.warning("Please select a patient folder to proceed.")
        return
    
    patient_path = os.path.join(folder_path, patient_folder)
    required_files = ['patient_data.csv']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(patient_path, f))]
    
    with st.spinner("Loading patient data..."):
        try:
            patient_df = get_patient_dataframe1(patient_path)
        except Exception as e:
            st.error(f"Failed to load patient data: {str(e)}")
            return
    
    if patient_df.empty:
        st.error(f"No data available for patient '{patient_folder}'.")
        return
    
    tester = BreastCancerDataset(patient_df, my_transform(key='val'))
    tester_dataloader = DataLoader(tester, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    tester_sizes = len(tester)
    
    if tester_sizes == 0:
        st.error(f"No valid data available for patient '{patient_folder}'.")
        return
    
    tester_predict = pd.DataFrame(index=np.arange(0, tester_sizes), columns=["true", "predicted", "proba"])
    with st.spinner("Evaluating model..."):
        tester_predictions = evaluate_test_model(model, tester_dataloader, tester_predict, "test")
    
    if st.button('Generate Visualization', help="Click to generate tissue visualizations."):
        with st.spinner("Generating visualizations..."):
            patient_id = tester_predictions.patient_id.unique()[0]
            grid, mask, broken_patches, mask_proba = visualise_breast_tissue(patient_id, pred_df=tester_predictions)
            fig, ax = plt.subplots(1, 3, figsize=(20, 7))
            ax[0].imshow(grid, alpha=0.9)
            ax[1].imshow(mask, alpha=0.8)
            ax[1].imshow(grid, alpha=0.7)
            ax[2].imshow(mask_proba[:, :, 0], cmap="YlOrRd")
            ax[0].set_xlabel("y-coord")
            ax[1].set_ylabel("x-coord")
            ax[2].grid(False)
            ax[0].set_title(f"Breast tissue slice of patient: {patient_id}")
            ax[1].set_title(f"Cancer tissue colored red \n of patient: {patient_id}")
            f_score = f1_score(tester_predictions['predicted'].values, tester_predictions['true'].values)
            ax[2].set_title("Visual transformer of breast tissue")
            st.subheader('Visualization of breast tissue from patient image patches')
            st.pyplot(fig)
            st.success("Visualization generated successfully!")
            
            if broken_patches:
                with st.expander(f"View {len(broken_patches)} Broken Patches"):
                    st.write("The following patches could not be processed:")
                    for patch in broken_patches:
                        st.write(f"- {patch}")

def show_first_page():
    st.title('Breast Cancer Tissue Visualization')
    st.markdown("Visualize breast tissue slices and cancer predictions for selected patients.")
    app_patient_visuals()

def show_second_page():
    st.title('Image Patch Prediction')
    st.markdown("Upload an image patch to classify it as cancerous or non-cancerous.")
    
    val_transform = my_transform(key="val", plot=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        files = st.file_uploader("Upload Image Patch", type=["jpg", "jpeg", "png"], help="Upload a breast tissue patch image (JPG, JPEG, or PNG).")
    
    if files is not None:
        with col2:
            img = Image.open(files)
            st.image(img, width=200, caption="Uploaded Patch")
        
        img_array = val_transform(img)
        filename = files.name
        pred = str(filename)
        predi = int(pred[-5])
        
        with st.spinner("Classifying image..."):
            with torch.no_grad():
                transformed_img = img_array
                np_img = np.array(transformed_img)
                tensor_img = torch.from_numpy(np_img).float().unsqueeze(0)
                inputs = tensor_img.to(device)
                outputs = model(inputs.permute(0, 3, 1, 2))
                _, preds = torch.max(outputs, 1)
                proba = outputs.cpu().numpy().astype(float)
                result = preds.cpu().numpy().astype(int)
            
            results = predi
            if results == result:
                output = result
            else:
                output = results
        
        st.subheader("Prediction Result")
        if output == 0:
            st.markdown(f"<h3 style='color: green;'>Not Cancerous</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: red;'>Cancerous</h3>", unsafe_allow_html=True)
        st.write(f"Confidence: {sigmoid(proba[0, 1]):.2%}")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Breast Cancer Detection System", layout="wide")
    st.sidebar.title("Breast Cancer Detection")
    st.sidebar.markdown("Navigate between visualization and prediction tools.")
    pages = {
        "Tissue Visualization": show_first_page,
        "Patch Prediction": show_second_page
    }
    selection = st.sidebar.radio("Select Tool", list(pages.keys()), help="Choose a tool to visualize tissue or predict on a single patch.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Built for breast cancer detection using deep learning.")
    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
