import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from torch.utils.data import Dataset
import glob
import math
import json
from torchsummary import summary
from torch.utils.data import DataLoader
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.ticker import MultipleLocator
import torchvision.models as models
from PIL import Image
import torch.nn.init as init
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
import random
import datetime
import threading 
import keyboard

# Set seed for random number generators
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 100
training_batch_size = 64
validation_batch_size = 64
test_batch_size = 64

############################################### Train, validation and test images with ESIs ########################################### 

# Define the list of folders and their respective CSV file 
training_folders = [      
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2021.3',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2021.3.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2021.11',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2021.11.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2021.12',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2021.12.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2022.1',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2022.1.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2022.2',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2022.2.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2022.3',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2022.3.csv',
            "selected": True  # Set to True to include in processing
        }, 
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2022.5',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2022.5.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2022.6',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2022.6.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2022.9',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2022.9.csv',
            "selected": True  # Set to True to include in processing
        }      
    ]
    
validation_folders = [   
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2023.2',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2023.2.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2023.7',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2023.7.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2023.9',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2023.9.csv',
            "selected": True  # Set to True to include in processing
        }
    ]
    
test_folders = [
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2021.2',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2021.2.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2022.10',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2022.10.csv',
            "selected": True  # Set to True to include in processing
        },
        {
            "folders_of_images_dir": r'C:\Maziar Mirsalehi\Folders of Images\2023.8',
            "csv_file_path": r'C:\Maziar Mirsalehi\Details of Data\Data information_2023.8.csv',
            "selected": True  # Set to True to include in processing
        }     
    ]

# Define a function to process images and apply ESI labels
def process_images_with_labels(folders_of_images_dir, csv_file_path):
    
    ESI_list = []
    images_path_list = []    
    eye_list = []    

    # Load the CSV file containing 3DV file names and ESIs
    csv_data = pd.read_csv(csv_file_path)
    
    # Iterate over the rows in the CSV file
    for index, row in csv_data.iterrows():
        
        # Extract the 3DV file name and ESI value from the current row
        file_name = row['3DV file name']
        esi = row['ESI']
        eye = row['Eye']
           
        # Construct the full path to the file within the specified root directory
        file_path = os.path.join(folders_of_images_dir, file_name)
        
        # Check if the file exists
        if os.path.exists(file_path):
            
            
                # Iterate over all images in the file
                for img_filename in os.listdir(file_path):
                    
                    # Construct the full image path
                    image_path = os.path.join(file_path, img_filename)
                    
                    # Append the image path to the list of image paths
                    images_path_list.append(image_path)

                # Append the corresponding ESI value to the label list 
                ESI_list.append(esi)
                eye_list.append(eye)


    return ESI_list, images_path_list, eye_list

def load_data(folders_info):
    ESI_list = []
    images_path_list = []
    eye_list = []
    for folder_info in folders_info:
        if folder_info["selected"]:
            folders_of_images_dir = folder_info["folders_of_images_dir"]
            csv_file_path = folder_info["csv_file_path"]
            # Process images and labels for the current folder
            folder_ESI_list, folder_images_path_list, folder_eye_list = process_images_with_labels(folders_of_images_dir, csv_file_path)
            # Concatenate the data from the current folder to the accumulated lists
            ESI_list += folder_ESI_list
            images_path_list += folder_images_path_list
            eye_list += folder_eye_list
    return ESI_list, images_path_list, eye_list


# Separate training, validation, and testing data loading
training_ESI_list, training_images_path_list, training_eye_list = load_data(training_folders)
validation_ESI_list, validation_images_path_list, validation_eye_list = load_data(validation_folders)
test_ESI_list, test_images_path_list, test_eye_list = load_data(test_folders)


############################################ CustomDataset class ##############################################################

class CustomDataset(Dataset):
    def __init__(self, images_path_list, ESI_list, eye_list, num_images_per_exam=16, crop_bottom=0.6, crop_left=0.25, crop_right=0.25):
        self.images_path_list = images_path_list
        self.ESI_list = ESI_list
        self.eye_list = eye_list
        self.num_images_per_exam = num_images_per_exam
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right

    def __getitem__(self, index):
        start_idx = index * self.num_images_per_exam
        end_idx = start_idx + self.num_images_per_exam
        exam_images_paths = self.images_path_list[start_idx:end_idx]
        exam_ESI = self.ESI_list[index]
        exam_eye = self.eye_list[index]                              # Use the label associated with the group of images

        # Load, resize, and crop the images
        stacked_exam_images = self.load_images(exam_images_paths)


        # Encode eye information into a tensor
        if exam_eye == "Right":
            eye_tensor = torch.tensor([1, 0], dtype=torch.float32)  # Right eye
        elif exam_eye == "Left":
            eye_tensor = torch.tensor([0, 1], dtype=torch.float32)  # Left eye
        

        return stacked_exam_images, exam_ESI, eye_tensor

    def __len__(self):
        return len(self.images_path_list) // self.num_images_per_exam

    def load_images(self, image_paths):
        exam_images = []
        for img_path in image_paths:
            # Load the image
            read_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Calculate cropping dimensions
            height, width = read_image.shape
            bottom_crop = int(self.crop_bottom * height)
            # Crop the image
            cropped_read_image = read_image[:-bottom_crop, 
                                      int(self.crop_left * width):width - int(self.crop_right * width)]
            # Resize the image
            resized_cropped_read_image = cv2.resize(cropped_read_image, (224, 224))
            exam_images.append(resized_cropped_read_image)

        # Concatenate along the last dimension to form a tensor with shape [224, 224, 16]
        stacked_exam_images = np.stack(exam_images, axis=0)

        return stacked_exam_images

####################################### Define the datasets for training, validation, and test ###################################

training_dataset = CustomDataset(training_images_path_list, training_ESI_list, training_eye_list)
validation_dataset = CustomDataset(validation_images_path_list, validation_ESI_list, validation_eye_list)
test_dataset = CustomDataset(test_images_path_list, test_ESI_list, test_eye_list)

############################################## Create data loaders ###############################################################

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=training_batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=validation_batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

############################################# Convolutional Neural Network Class #####################################################

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()

        # Load the DenseNet-121 model without pretrained weights
        self.densenet = models.densenet121(pretrained=False)
        
        # Modify the first convolutional layer to accept 16-channel input
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels=16, 
            out_channels=self.densenet.features.conv0.out_channels,
            kernel_size=self.densenet.features.conv0.kernel_size,
            stride=self.densenet.features.conv0.stride,
            padding=self.densenet.features.conv0.padding,
            bias=self.densenet.features.conv0.bias
        )
        
        # Get the number of features for the classifier layer
        num_features = self.densenet.classifier.in_features

        # Modify the classifier layer for a single output
        self.densenet.classifier = nn.Linear(num_features, 1)

        # Additional fully connected layer to process the combined features
        self.final_fc = nn.Linear(3, 1)  # 1 feature from DenseNet + 2 for the eye parameters

    def forward(self, x, eye):
        batch_size = x.size(0)

        # Process the image through the DenseNet model
        x = self.densenet(x)

        # Ensure the eye tensor is the right shape
        eye = eye.view(batch_size, -1)  # Reshape to [batch_size, 2]

        # Concatenate the eye parameter with the output from DenseNet
        combined = torch.cat((x, eye), dim=1)  # Shape: [batch_size, 3]

        # Process the combined features through the final fully connected layer
        out = self.final_fc(combined)

        return out

# Example instantiation
model = DenseNet121()
print(model)

# Calculate the total number of parameters that are not frozen
total_params = sum(p.numel() for p in model.parameters())

# Print the total number of non-frozen parameters
print("Total number of parameters:", total_params)

# Define the optimizer for the parameters of the model
optimizer= torch.optim.AdamW(model.parameters(), lr= 0.01, weight_decay=0.05)

# Define the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Initialize lists to store losses for each epoch
training_mse_epochs = []
training_mae_epochs = []
training_rmse_epochs = []
validation_mse_epochs = []
validation_mae_epochs = []
validation_rmse_epochs = []

# Define a function to format the time in Hour:Minute:Second
def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{:02}:{:02}:{:02}".format(int(h), int(m), int(s))

# Record the start time before the training loop
start_time_total = time.time()

# Initialize the previous learning rate variable
prev_lr = None

for epoch in range(num_epochs):

    start_time = time.time()                           # Record the start time
    training_mse_losses = []
    training_mae_losses = []
    training_rmse_losses = []
    validation_mse_losses = []
    validation_mae_losses = []
    validation_rmse_losses = []

    # Training loop
    model.train()

    for batch_idx, (images, labels, eyes) in enumerate(training_loader):
        images = images.to(device).float()
        labels = labels.to(device).float()
        eyes = eyes.to(device)

        # Forward pass
        outputs = model(images, eyes)

        # Use Mean Absolute Error 
        mse_loss = F.mse_loss(outputs.squeeze(), labels.float().squeeze())

        # Calculate MAE loss
        mae_loss = F.l1_loss(outputs.squeeze(), labels.float().squeeze())
        
        # Calculate RMSE
        rmse_loss = np.sqrt(mse_loss.item())

        # Append the current losses to different training losses
        training_mse_losses.append(mse_loss.item())    
        training_mae_losses.append(mae_loss.item())
        training_rmse_losses.append(rmse_loss.item())    

        # Backward and optimize
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        # Perform a garbage collection to free any unused memory
        torch.cuda.empty_cache()
    
    # Calculate average training loss for the epoch
    training_MSE = sum(training_mse_losses) / len(training_mse_losses)
    training_MAE = sum(training_mae_losses) / len(training_mae_losses)
    training_RMSE = sum(training_rmse_losses) / len(training_rmse_losses)

    # Validation loop
    model.eval()

    with torch.no_grad():
        for images, labels, eyes in validation_loader:
            
            images = images.to(device).float()
            labels = labels.to(device).float()
            eyes = eyes.to(device)

            outputs = model(images, eyes)

            mse_loss = F.mse_loss(outputs.squeeze(), labels.float().squeeze())
            mae_loss = F.l1_loss(outputs.squeeze(), labels.float().squeeze())
            rmse_loss = np.sqrt(mse_loss.item())

            validation_mse_losses.append(mse_loss.item()) 
            validation_mae_losses.append(mae_loss.item())
            validation_rmse_losses.append(rmse_loss.item())

            # Perform a garbage collection to free any unused memory
            torch.cuda.empty_cache()

    # Calculate average validation loss for the epoch
    validation_MSE = sum(validation_mse_losses) / len(validation_mse_losses)
    validation_MAE = sum(validation_mae_losses) / len(validation_mae_losses)
    validation_RMSE = sum(validation_rmse_losses) / len(validation_rmse_losses)
    
    # Update the learning rate scheduler based on validation loss
    scheduler.step(validation_MSE)

    # Access the current learning rate
    current_lr = "{:.8f}".format(scheduler.get_last_lr()[0])

    # Check if the learning rate has changed
    if prev_lr is not None and current_lr != prev_lr:
        print(f'Learning rate changed: {prev_lr} --> {current_lr}')
    
    prev_lr = current_lr  # Update the previous learning rate for the next epoch

    end_time = time.time()                                 # Record the end time
    epoch_time = end_time - start_time

    print(f'Epoch [{epoch+1}/{num_epochs}]       Training MSE: {training_MSE:.5f}      Validation MSE: {validation_MSE:.5f}    Training MAE: {training_MAE:.5f}      Validation MAE: {validation_MAE:.5f}   Training RMSE: {training_RMSE:.5f}     Validation RMSE: {validation_RMSE:.5f}    Time: {format_time(epoch_time)}')

    # Append the average training and validation losses for the epoch
    training_mse_epochs.append(training_MSE)
    training_mae_epochs.append(training_MAE)
    training_rmse_epochs.append(training_RMSE)
    validation_mse_epochs.append(validation_MSE)
    validation_mae_epochs.append(validation_MAE)
    validation_rmse_epochs.append(validation_RMSE)

    # Get the current date
    current_date = datetime.datetime.now().strftime("%d.%m.%Y")

    model_save_path = rf'C:\Maziar Mirsalehi\Python codes\CNNs\DenseNet121 model of epoch\DenseNet121_{epoch + 1}_{current_date}.pth'
    torch.save(model, model_save_path)
    print(f'Model state saved for epoch {epoch + 1} at {model_save_path}')


# Calculate and print the total training time
end_time_total = time.time()
total_training_time = end_time_total - start_time_total
print(f'Total Time: {format_time(total_training_time)}')

####################################################### Test evaluation ########################################################

model.eval()  

predictions = []

start_time = time.time()                  # Record the start time

test_mse_losses = [] 
test_mae_losses = []
test_rmse_losses = [] 
    
with torch.no_grad():
        
    for images, labels, eyes in test_loader:
        images = images.to(device).float()
        labels = labels.to(device).float()
        eyes = eyes.to(device)

        outputs = model(images, eyes)

        # Append predictions and labels to the list
        for i in range(len(outputs)):
            predictions.append([labels[i].item(), outputs[i].item()])

        mse_loss = F.mse_loss(outputs.squeeze(), labels.float().squeeze())
        mae_loss = F.l1_loss(outputs.squeeze(), labels.float().squeeze())
        rmse_loss = np.sqrt(mse_loss.item())

        test_mse_losses.append(mse_loss.item())
        test_mae_losses.append(mae_loss.item())
        test_rmse_losses.append(rmse_loss.item())
        
        # Perform a garbage collection to free any unused memory
        torch.cuda.empty_cache()


test_MSE = sum(test_mse_losses) / len(test_mse_losses)
test_MAE = sum(test_mae_losses) / len(test_mae_losses)
test_RMSE = sum(test_rmse_losses) / len(test_rmse_losses)

end_time = time.time()                          # Record the end time 
test_time = end_time - start_time
print(f'Test MSE: {test_MSE:.5f}        Test MAE: {test_MAE:.5f}       Test RMSE: {test_RMSE:.5f}      Test Time: {format_time(test_time)}')

#Save predictions to a CSV file
current_date = datetime.datetime.now().strftime("%d.%m.%Y")
predictions_df = pd.DataFrame(predictions, columns=['Normalised ESI', 'Predictions'])
predictions_df.to_csv(rf'C:\Maziar Mirsalehi\Results\DenseNet121\DenseNet121_Test predictions_{current_date}.csv', index=False)

######################################### Saving Training, Validation and Test MSE on a csv file #######################################

# Create a DataFrame to hold the MSE values
mse_data = {
    'Training MSE': training_mse_epochs,
    'Validation MSE': validation_mse_epochs,
    'Test MSE': test_MSE,
    'Training MAE': training_mae_epochs,
    'Validation MAE': validation_mae_epochs,
    'Test MAE': test_MAE,
    'Training RMSE': training_rmse_epochs,
    'Validation RMSE': validation_rmse_epochs,
    'Test RMSE': test_RMSE 
}

# Convert the dictionary to a DataFrame
mse_df = pd.DataFrame(mse_data)


# Save the DataFrame to a CSV file
current_date = datetime.datetime.now().strftime("%d.%m.%Y")
mse_df.to_csv(rf'C:\Maziar Mirsalehi\Results\DenseNet121\DenseNet121_Values_{current_date}.csv', index=False)

#######################################################################################################################################

# Save the entire model state dictionary
current_date = datetime.datetime.now().strftime("%d.%m.%Y")
torch.save(model, rf'C:\Maziar Mirsalehi\Python codes\CNNs\DenseNet121 model\DenseNet121_model_{current_date}.pth')


