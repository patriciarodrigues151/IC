import os

import matplotlib.pyplot as plt
import numpy as np

import librosa
import librosa.display

import os
import torch
import torchvision
from torchvision import datasets, transforms


def createSpectrogramsDataBase():
    # Specify the directory containing the audio files
    audio_dir = 'C:/Users/patri/Desktop/IC/cantos'
    
    # Get a list of all the audio files in the directory
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            audio_files.append(file)
    
    # Check if any audio files were found
    if len(audio_files) == 0:
        print("No audio files found in the specified directory.")
        return

    # Create a directory to save the images
    if not os.path.exists('spectrograms'):
       os.makedirs('spectrograms')
       
    #loop through the audio files
    for file in audio_files:
        audio_path = os.path.join(audio_dir, file)
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            # Create a new directory for each audio file
            if not os.path.exists(f'spectrograms/{file}'):
                os.makedirs(f'spectrograms/{file}')
            # Divide the audio into 10-second segments
            for i in range(0, len(y), sr*10):
                segment = y[i:i+sr*10]
                S = librosa.stft(segment, n_fft=2048, hop_length=512, win_length=1024)
                S = np.abs(S)
                spectrogram = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', sr=sr)
                plt.axis('off')
                plt.savefig(f'spectrograms/{file}/spectrogram_{i//sr}.png', bbox_inches='tight', pad_inches=0)
                plt.clf()
            print(f"Processing {file} - OK")
        except:
            print(f"Error processing {file}. Skipping.")

def inputSpectrogramsDataBase():
    # Set the base path where the bird folders are located
    base_path = 'C:/Users/patri/Desktop/IC/spectrograms'

    # Loop through each bird folder
    for bird_folder in os.listdir(base_path):
        bird_folder_path = os.path.join(base_path, bird_folder)
        if os.path.isdir(bird_folder_path):
            # Loop through each spectrogram image inside the bird folder
            for image_file in os.listdir(bird_folder_path):
                image_file_path = os.path.join(bird_folder_path, image_file)
                if image_file.endswith('.png'):
                    # Load the spectrogram image
                    img = cv2.imread(image_file_path)
                    
                    # Display the image
                    plt.imshow(img)
                    plt.show()

def neuralNetwork():

    data_dir = r'C:\Users\patri\Desktop\IC\spectrograms'

    ### Define a data transformer for preprocessing
    data_transformer = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create a dataset from the spectrogram images appling the transformation to images
    dataset = datasets.ImageFolder(data_dir, transform=data_transformer)

    # Print the length of the dataset
    print("Dataset length:", len(dataset))

    # Loop through the dataset and print the image and label for each sample
    for i, (image, label) in enumerate(dataset):
        print("Sample", i, "Image shape:", image.shape, "Label:", label)
        if i == 5:
            break





    # Split the dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])

    # Create a dataloader to load the data in batches
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Define the Convolutional Neural Network (CNN) architecture
    model = torchvision.models.resnet18(pretrained=False, num_classes=len(dataset.classes))

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train the model
    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch: {epoch+1} Loss: {running_loss/len(train_dataloader)}')

    # Evaluate the model on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_dataloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on validation set: {100*correct/total}%')


def main():
    createSpectrograms()
    #inputSpectrogramsDataBase()
    #neuralNetwork()
    

main()
