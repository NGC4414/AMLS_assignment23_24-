from sklearn.linear_model import LogisticRegression
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras import layers, models
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.utils import to_categorical
from skimage.util import montage
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import (Input, Dense, Dropout, Activation, GlobalAveragePooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D)
from keras.models import save_model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score


def load_data_pneumonia():
    path = r"Datasets\pneumoniamnist.npz"
    with np.load(path) as pneumoniamnist:
        x_train = pneumoniamnist['train_images']
        y_train = pneumoniamnist['train_labels']
        x_val = pneumoniamnist['val_images']
        y_val = pneumoniamnist['val_labels']
        x_test = pneumoniamnist['test_images']
        y_test = pneumoniamnist['test_labels']

    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))



####################Data pre-processing functions########################################

def visualize_images(images, labels, n_channels, length=20):
    scale = length * length

    # Create an array to store the images
    image = np.zeros((scale, 28, 28, 3)) if n_channels == 3 else np.zeros((scale, 28, 28))
    
    # Create an array of indices for shuffling
    index = [i for i in range(scale)]
    np.random.shuffle(index)
    
    plt.figure(figsize=(6, 6))

    for idx in range(scale):
        img = images[idx]
        if n_channels == 3:
            img = img.reshape(28, 28, n_channels)
        else:
            img = img.reshape(28, 28)
        image[index[idx]] = img

    if n_channels == 1:
        image = image.reshape(scale, 28, 28)
        arr_out = montage(image)
        plt.imshow(arr_out, cmap='gray')
    else:
        image = image.reshape(scale, 28, 28, 3)
        arr_out = montage(image, multichannel=3)
        plt.imshow(arr_out)

    plt.title("Sample Images")
    plt.axis("off")
    plt.show()


def plot_class_distribution(y_train, y_val, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    datasets = [y_train, y_val, y_test]
    titles = ['Training Set', 'Validation Set', 'Test Set']
    colors = [['blue', 'orange'], ['green', 'red'], ['purple', 'brown']]

    for i, (dataset, title, color) in enumerate(zip(datasets, titles, colors)):
        classes, counts = np.unique(dataset, return_counts=True)
        axes[i].bar(classes, counts, color=color)
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Number of Images')
        axes[i].set_xticks(classes)
        axes[i].set_xticklabels(['Normal', 'Pneumonia'])
        axes[i].set_title(title)

    plt.tight_layout()
    plt.show()


def analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test, images, n_channels, length=20):
    ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()
    # First, plot data distribution
    plot_class_distribution(y_train, y_val, y_test)

    # Then, visualize sample images
    visualize_images(images, y_train, n_channels, length)

# Usage
#((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()
#analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test, x_train, n_channels=1, length=20)
    



def pneumoniaLogRegrPredict(x_train, y_train, x_val, y_val, x_test, y_test):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)

    # Flatten images to 1-D vector of 784 features (28*28)
    train_images_flatten = np.reshape(x_train, (len(x_train), -1))
    val_images_flatten = np.reshape(x_val, (len(x_val), -1))
    test_images_flatten = np.reshape(x_test, (len(x_test), -1))
    #print(train_images_flatten.shape)
    print(test_images_flatten.shape)
    print(y_test.shape)
    # if ravel() not applied this error shows
    #DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the 
    # shape of y to (n_samples, ), for example using ravel(). y = column_or_1d(y, warn=True)

     # Flatten the labels
    y_train_flatten = y_train.ravel()
    y_val_flatten = y_val.ravel()
    y_test_flatten = y_test.ravel()

    # Standardize features
    scaler = StandardScaler()
    train_images_scaled = scaler.fit_transform(train_images_flatten)
    val_images_scaled = scaler.transform(val_images_flatten)
    test_images_scaled = scaler.transform(test_images_flatten)
    # Train the model on the training data
    model.fit(train_images_scaled, y_train_flatten)

    # Validate the model on the validation data
    val_accuracy = model.score(val_images_scaled, y_val_flatten)
    print(f'Validation Accuracy: {val_accuracy}')

    # Test the model on the test data
    test_images_scaled = scaler.transform(np.reshape(x_test, (len(x_test), -1)))
    test_accuracy = model.score(test_images_scaled, y_test_flatten)
    print(f'Test Accuracy: {test_accuracy}')

    # Make predictions on the test set
    y_pred = model.predict(test_images_scaled)

    # Print the predictions and evaluation metrics
    print("Predictions:", y_pred)
    print("Confusion Matrix:\n", confusion_matrix(y_test_flatten, y_pred))
    print("Accuracy on test set:", accuracy_score(y_test_flatten, y_pred))
    print("Classification Report:\n", classification_report(y_test_flatten, y_pred))




def preprocess_pneumoniamnist(x_train, x_val, x_test, y_train, y_test, y_val, num_classes):
    # Convert to float32
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    # Rescale values to [0, 1]
    x_train /= 255
    x_val /= 255
    x_test /= 255

    # Expand dimensions
    x_train = np.expand_dims(x_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # One-hot encode labels
    y_train_onehot = to_categorical(y_train, num_classes)
    y_val_onehot = to_categorical(y_val, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)

    return x_train, x_val, x_test, y_train_onehot, y_val_onehot, y_test_onehot


def train_and_save_cnn(x_train, y_train, x_val, y_val, x_test, y_test):

    x_train, x_val, x_test, y_train_onehot, y_val_onehot, y_test_onehot = preprocess_pneumoniamnist(x_train, x_val, x_test, y_train, y_test, y_val, num_classes=2)

   
    print('y_train_onehot shape:', y_train_onehot.shape)
    print('y_test_onehot shape:', y_test_onehot.shape)
    print('y_val_onehot shape:', y_val_onehot.shape)
    input = Input(shape=x_train.shape[1:])

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='last_conv_layer')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    
    output = Dense(units=2, activation='softmax', name='predictions')(x)

    model = Model(inputs=[input], outputs=[output])
    print(model.summary())
    

    batch_size = 256
    epochs = 20
    init_lr = 0.001
    opt = Adam(learning_rate=init_lr)
    model.compile(optimizer = opt, loss='categorical_crossentropy', metrics='accuracy')
    cnn_history = model.fit(x_train, y_train_onehot,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_val, y_val_onehot),
                  verbose=2)
    
     # Save the trained model
    model.save('A/pretrained_model.h5')
 
    return model, cnn_history


#((x_train, y_train), (x_val, y_val), (x_test, y_test))= load_data_pneumonia()
#trained_model, history = train_and_save_cnn(x_train, y_train, x_val, y_val, x_test, y_test)



def pneumoniaCNNPredict(x_test, y_test):
    model = load_model('A/pretrained_model.h5')

    # Preprocess the test data
    x_test = x_test.astype('float32') / 255
    x_test = np.expand_dims(x_test, axis=3)
    y_test_onehot = to_categorical(y_test)

    # Make predictions
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy and recall
    accuracy = accuracy_score(y_test_onehot.argmax(axis=1), predicted_classes)
    recall = recall_score(y_test_onehot.argmax(axis=1), predicted_classes)

    # Print the predictions and evaluation metrics
    print("Predictions:", predicted_classes)
    print("Accuracy on test set:", accuracy)
    print("Recall on test set:", recall)

    # Create a confusion matrix
    cm = confusion_matrix(y_test_onehot.argmax(axis=1), predicted_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                yticklabels=['True Normal', 'True Pneumonia'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Create a bar plot for accuracy and recall
    metrics = ['Accuracy', 'Recall']
    values = [accuracy, recall]
    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=['blue', 'orange'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Evaluation Metrics')
    plt.show()



   




