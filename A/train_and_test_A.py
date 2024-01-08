import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers, models
from keras.layers import (Input, Dense, Dropout, Activation, GlobalAveragePooling2D, 
                          BatchNormalization, Flatten, Conv2D, MaxPooling2D)
from keras.models import Model, save_model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from skimage.util import montage

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            recall_score, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle, class_weight

import tensorflow as tf

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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


###Logistic regression
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
    print("Accuracy on test set:", accuracy_score(y_test_flatten, y_pred))
    print("Classification Report:\n", classification_report(y_test_flatten, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_flatten, y_pred))


# def train_and_evaluate_svms(x_train, y_train, x_val, y_val, x_test, y_test):
#     kernels = ['linear', 'rbf', 'poly', 'sigmoid']
#     models = {}
#     scalers = {}

#     for kernel in kernels:
#         # Standardize the data
#         scaler = StandardScaler()
#         x_train_scaled = scaler.fit_transform(x_train.reshape((x_train.shape[0], -1)))
#         x_val_scaled = scaler.transform(x_val.reshape((x_val.shape[0], -1)))

#         # Train the SVM
#         svm_model = SVC(kernel=kernel, probability=True)
#         svm_model.fit(x_train_scaled, y_train.ravel())

#         # Store the model and scaler
#         models[kernel] = svm_model
#         scalers[kernel] = scaler

#         # Validate the model
#         val_predictions = svm_model.predict(x_val_scaled)
#         val_accuracy = accuracy_score(y_val, val_predictions)
#         print(f"Validation Accuracy with {kernel} kernel: {val_accuracy}")

#         # Evaluate on test data
#         x_test_scaled = scaler.transform(x_test.reshape((x_test.shape[0], -1)))
#         test_predictions = svm_model.predict(x_test_scaled)
#         test_accuracy = accuracy_score(y_test, test_predictions)
#         test_roc_auc = roc_auc_score(y_test, svm_model.predict_proba(x_test_scaled)[:, 1])
#         print(f"Test Accuracy with {kernel} kernel: {test_accuracy}")
#         print(f"Test ROC-AUC with {kernel} kernel: {test_roc_auc}")
#         print(classification_report(y_test, test_predictions))

#         # Plotting the confusion matrix
#         cm = confusion_matrix(y_test, test_predictions)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
#         plt.title(f'Confusion Matrix for {kernel} Kernel')
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         plt.show()

#     return models, scalers
    
def train_and_evaluate_svms(x_train, y_train, x_val, y_val, x_test, y_test):
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    models = {}
    scalers = {}
    confusion_matrices = {}

    for kernel in kernels:
        # Standardize the data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train.reshape((x_train.shape[0], -1)))
        x_val_scaled = scaler.transform(x_val.reshape((x_val.shape[0], -1)))

        # Train the SVM
        svm_model = SVC(kernel=kernel, probability=True)
        svm_model.fit(x_train_scaled, y_train.ravel())

        # Store the model and scaler
        models[kernel] = svm_model
        scalers[kernel] = scaler

        # Validate the model
        val_predictions = svm_model.predict(x_val_scaled)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy with {kernel} kernel: {val_accuracy}")

        # Evaluate on test data
        x_test_scaled = scaler.transform(x_test.reshape((x_test.shape[0], -1)))
        test_predictions = svm_model.predict(x_test_scaled)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_roc_auc = roc_auc_score(y_test, svm_model.predict_proba(x_test_scaled)[:, 1])
        print(f"Test Accuracy with {kernel} kernel: {test_accuracy}")
        print(f"Test ROC-AUC with {kernel} kernel: {test_roc_auc}")
        print(classification_report(y_test, test_predictions))

        # Compute and store the confusion matrix
        confusion_matrices[kernel] = confusion_matrix(y_test, test_predictions)

    # Plotting all confusion matrices
    fig, axes = plt.subplots(1, len(kernels), figsize=(20, 5))
    for i, kernel in enumerate(kernels):
        sns.heatmap(confusion_matrices[kernel], annot=True, fmt='g', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {kernel} Kernel')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.show()

    return models, scalers


def plot_roc_curves(models, scalers, x_test, y_test):
    plt.figure(figsize=(10, 8))

    for kernel in models:
        model = models[kernel]
        scaler = scalers[kernel]

        # Scale test data
        x_test_scaled = scaler.transform(x_test.reshape((x_test.shape[0], -1)))

        # Compute predicted probabilities
        y_pred_prob = model.predict_proba(x_test_scaled)[:, 1]

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.plot(fpr, tpr, label=f'ROC curve (kernel={kernel}, area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


####SVM using PCA


# def train_and_evaluate_svms_pca(x_train, y_train, x_val, y_val, x_test, y_test, n_components=0.95):
#     kernels = ['linear', 'rbf', 'poly', 'sigmoid']
#     models = {}
#     scalers = {}
#     pcas = {}

#     for kernel in kernels:
#         # Standardize the data
#         scaler = StandardScaler()
#         x_train_scaled = scaler.fit_transform(x_train.reshape((x_train.shape[0], -1)))
#         x_val_scaled = scaler.transform(x_val.reshape((x_val.shape[0], -1)))

#         # Apply PCA
#         pca = PCA(n_components=n_components)
#         x_train_pca = pca.fit_transform(x_train_scaled)
#         x_val_pca = pca.transform(x_val_scaled)

#         # Train the SVM
#         svm_model = SVC(kernel=kernel, probability=True)
#         svm_model.fit(x_train_pca, y_train.ravel())

#         # Store the model, scaler, and PCA
#         models[kernel] = svm_model
#         scalers[kernel] = scaler
#         pcas[kernel] = pca

#         # Validate the model
#         val_predictions = svm_model.predict(x_val_pca)
#         val_accuracy = accuracy_score(y_val, val_predictions)
#         print(f"Validation Accuracy with {kernel} kernel: {val_accuracy}")

#         # Evaluate on test data
#         x_test_scaled = scaler.transform(x_test.reshape((x_test.shape[0], -1)))
#         x_test_pca = pca.transform(x_test_scaled)
#         test_predictions = svm_model.predict(x_test_pca)
#         test_accuracy = accuracy_score(y_test, test_predictions)
#         test_roc_auc = roc_auc_score(y_test, svm_model.predict_proba(x_test_pca)[:, 1])
#         print(f"Test Accuracy with {kernel} kernel: {test_accuracy}")
#         print(f"Test ROC-AUC with {kernel} kernel: {test_roc_auc}")
#         print(classification_report(y_test, test_predictions))

#         # Plotting the confusion matrix
#         cm = confusion_matrix(y_test, test_predictions)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
#         plt.title(f'Confusion Matrix for {kernel} Kernel (with PCA)')
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         plt.show()

#     return models, scalers, pcas

def train_and_evaluate_svms_pca(x_train, y_train, x_val, y_val, x_test, y_test, n_components=0.95):
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    models = {}
    scalers = {}
    pcas = {}
    confusion_matrices = {}

    for kernel in kernels:
        # Standardize the data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train.reshape((x_train.shape[0], -1)))
        x_val_scaled = scaler.transform(x_val.reshape((x_val.shape[0], -1)))

        # Apply PCA
        pca = PCA(n_components=n_components)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_val_pca = pca.transform(x_val_scaled)

        # Train the SVM
        svm_model = SVC(kernel=kernel, probability=True)
        svm_model.fit(x_train_pca, y_train.ravel())

        # Store the model, scaler, and PCA
        models[kernel] = svm_model
        scalers[kernel] = scaler
        pcas[kernel] = pca

        # Validate the model
        val_predictions = svm_model.predict(x_val_pca)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy with {kernel} kernel: {val_accuracy}")

        # Evaluate on test data
        x_test_scaled = scaler.transform(x_test.reshape((x_test.shape[0], -1)))
        x_test_pca = pca.transform(x_test_scaled)
        test_predictions = svm_model.predict(x_test_pca)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_roc_auc = roc_auc_score(y_test, svm_model.predict_proba(x_test_pca)[:, 1])
        print(f"Test Accuracy with {kernel} kernel: {test_accuracy}")
        print(f"Test ROC-AUC with {kernel} kernel: {test_roc_auc}")
        print(classification_report(y_test, test_predictions))

        # Compute and store the confusion matrix
        confusion_matrices[kernel] = confusion_matrix(y_test, test_predictions)

    # Plotting all confusion matrices
    fig, axes = plt.subplots(1, len(kernels), figsize=(20, 5))
    for i, kernel in enumerate(kernels):
        sns.heatmap(confusion_matrices[kernel], annot=True, fmt='g', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {kernel} Kernel (with PCA)')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.show()

    return models, scalers, pcas


def plot_roc_curves_pca(models, scalers, pcas, x_test, y_test):
    plt.figure(figsize=(10, 8))

    # For each kernel type
    for kernel in models:
        model = models[kernel]
        scaler = scalers[kernel]
        pca = pcas[kernel]  # Retrieve the corresponding PCA object

        # Scale and then apply PCA transformation to test data
        x_test_scaled = scaler.transform(x_test.reshape((x_test.shape[0], -1)))
        x_test_pca = pca.transform(x_test_scaled)  # Apply PCA transformation

        # Compute predicted probabilities
        y_pred_prob = model.predict_proba(x_test_pca)[:, 1]

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.plot(fpr, tpr, label=f'ROC curve (kernel={kernel}, area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    

###CNN

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
    print('x_train shape:',x_train.shape)
    print('x_test shape:',x_test.shape)
    print('x_val shape:',x_val.shape)

    # # One-hot encode labels
    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)
    y_val_onehot = to_categorical(y_val)

    print('y_train_onehot shape:', y_train_onehot.shape)
    print('y_test_onehot shape:', y_test_onehot.shape)
    print('y_val_onehot shape:', y_val_onehot.shape)
     
    return x_train, x_val, x_test, y_train_onehot, y_val_onehot, y_test_onehot

def train_and_save_cnn(x_train, y_train, x_val, y_val, x_test, y_test):

    x_train, x_val, x_test, y_train_onehot, y_val_onehot, y_test_onehot = preprocess_pneumoniamnist(x_train, x_val, x_test, y_train, y_test, y_val, num_classes=2)

    input = Input(shape=x_train.shape[1:])
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='last_conv_layer')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    output = Dense(2, activation='softmax', name='predictions')(x)

    model = Model(inputs=[input], outputs=[output])
    print(model.summary())

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_history = model.fit(x_train, y_train_onehot,
                            batch_size=128,
                            epochs=20,
                            validation_data=(x_val, y_val_onehot),
                            verbose=2)

    model.save('A/pretrained_model.h5')
    return model, cnn_history

    

def plot_training_history(history):
    # Plotting the training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')
    plt.show()

def pneumoniaCNNPredict(x_test, y_test):
    model = load_model('A/pretrained_model.h5')

    # Preprocess the test data
    x_test = x_test.astype('float32') / 255
    x_test = np.expand_dims(x_test, axis=3)
    y_test_onehot = to_categorical(y_test)
    

    # Make predictions
    predicted_probs = model.predict(x_test) 
    print(predicted_probs.shape)
    predicted_classes = np.argmax(predicted_probs, axis=1) 
    model.evaluate(x_test, y_test_onehot, verbose=2)

    classes = ['normal', 'pneumonia']
    print(classification_report(y_test, predicted_classes, target_names=classes))

    # Calculate accuracy and recall
    accuracy = accuracy_score(y_test, predicted_classes)
    recall = recall_score(y_test, predicted_classes)

    # Extract probabilities for the positive class (assuming class '1' is pneumonia)
    predicted_probs_positive = predicted_probs[:, 1]

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, predicted_probs_positive)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Print the predictions, evaluation metrics, and AUC
    print("Predictions:", predicted_classes)
    print("Accuracy on test set:", accuracy)
    print("Recall on test set:", recall)
    print("AUC on test set:", roc_auc)

    # Create a confusion matrix
    cm = confusion_matrix(y_test, predicted_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                yticklabels=['True Normal', 'True Pneumonia'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
