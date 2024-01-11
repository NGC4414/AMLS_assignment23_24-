# Data processing and visualization
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Keras for model building
from keras.models import Sequential, Model, load_model
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, 
                          GlobalAveragePooling2D, ReLU, Dropout, BatchNormalization)
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Optimizers and loss functions
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2

# Metrics and TensorFlow
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
import tensorflow as tf

####Focal Loss
def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)
    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha*(1-p_t)^gamma*log(p_t)
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow((1 - y_pred), gamma)
        
        # Calculate focal loss
        loss = weight * cross_entropy
        
        # Sum the losses in mini_batch
        loss = tf.reduce_sum(loss, axis=1)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def load_data_pathmnist():
    path = r"Datasets\pathmnist.npz"
    with np.load(path) as pathmnistmnist:
        x_train = pathmnistmnist['train_images']
        y_train = pathmnistmnist['train_labels']
        x_val = pathmnistmnist['val_images']
        y_val = pathmnistmnist['val_labels']
        x_test = pathmnistmnist['test_images']
        y_test = pathmnistmnist['test_labels']

    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))

((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pathmnist()

# Dictionary that maps class labels to their names
class_label_names = {
    '0': 'adipose',
    '1': 'background',
    '2': 'debris',
    '3': 'lymphocytes',
    '4': 'mucus',
    '5': 'smooth muscle',
    '6': 'normal colon mucosa',
    '7': 'cancer-associated stroma',
    '8': 'colorectal adenocarcinoma epithelium'
}

# Function to print class label descriptions 
def print_class_label_descriptions(class_label_names):
    print("Class Label Descriptions:")
    for label, name in class_label_names.items():
        print(f"Class {label}: {name}")


def plot_class_distribution_pathmnist(y_train, y_val, y_test):
    print_class_label_descriptions(class_label_names)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    datasets = [y_train, y_val, y_test]
    titles = ['Training Set', 'Validation Set', 'Test Set']
    colors = ['skyblue', 'salmon', 'palegreen']  
    for i, (dataset, title, color) in enumerate(zip(datasets, titles, colors)):
        classes, counts = np.unique(dataset, return_counts=True)
        axes[i].bar(classes, counts, color=color)
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Number of Images')
        axes[i].set_xticks(classes)
        axes[i].set_title(f"{title} (n={len(dataset)})")  # Show the number of samples in the title
    plt.tight_layout()
    plt.show()


def show_sample_images(x_data, y_data, class_label_names, num_samples_per_class=3, num_classes=9):
    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(15, 10))
    for class_label in range(num_classes):
        class_indices = np.where(y_data == class_label)[0]
        sampled_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)

        for sample_idx, data_idx in enumerate(sampled_indices):
            image = x_data[data_idx].reshape(28, 28, 3)  # Assuming RGB images
            axes[class_label, sample_idx].imshow(image)
            axes[class_label, sample_idx].axis('off')
            axes[class_label, sample_idx].set_title(class_label_names[str(class_label)])

    plt.tight_layout()
    plt.show()

####VGG

def preprocess_data(x_train, y_train, x_val, y_val):
    # Normalize images
    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255

    # Convert labels to one-hot encoding
    y_train_onehot = to_categorical(y_train, num_classes=9)
    y_val_onehot = to_categorical(y_val, num_classes=9)

    return x_train, y_train, x_val, y_val, y_train_onehot, y_val_onehot

    
def create_vgg(input_shape, hidden_units, output_shape):
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
    
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Conv Block 2
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='same'))
    model.add(ReLU())
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    # Classifier
    model.add(Flatten())
    model.add(Dense(output_shape, activation='softmax'))

    return model


def train_model_vgg_cross(model, x_train, y_train, x_val, y_val, learning_rate=0.001, momentum=0.9, epochs=50, batch_size=128, patience=5, model_path='B/vgg_no_aug_crossentropy.h5'):
    x_train, y_train, x_val, y_val, y_train_onehot, y_val_onehot = preprocess_data(x_train, y_train, x_val, y_val)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')
    callbacks_list = [model_checkpoint, early_stopping]

    # Training the model
    history = model.fit(x_train, y_train_onehot,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val_onehot),
                        callbacks=callbacks_list)
  
    return history

from keras.preprocessing.image import ImageDataGenerator

def train_model_vgg_with_augmentation_cross(model, x_train, y_train, x_val, y_val, learning_rate=0.001,  epochs=50, batch_size=128, patience=5, model_path='B/vgg_with_aug_crossentropy.h5'):
    x_train, y_train, x_val, y_val, y_train_onehot, y_val_onehot = preprocess_data(x_train, y_train, x_val, y_val)
    custom_focal_loss = focal_loss(gamma=2., alpha=4.)

    # Define the data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Create a training data generator
    train_generator = train_datagen.flow(x_train, y_train_onehot, batch_size=batch_size)
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=CategoricalCrossentropy(from_logits=False),
                  #loss=custom_focal_loss,
                  metrics=['accuracy'])
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')
    callbacks_list = [model_checkpoint, early_stopping]
    # Training the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(x_train) // batch_size,  # Number of steps per epoch
        validation_data=(x_val, y_val_onehot),
        callbacks=callbacks_list
    )
    return history


def train_model_vgg_focal(model, x_train, y_train, x_val, y_val, learning_rate=0.001, epochs=50, batch_size=128, patience=5, model_path= 'B/vgg_no_aug_focal_loss.h5'):
    x_train, y_train, x_val, y_val, y_train_onehot, y_val_onehot = preprocess_data(x_train, y_train, x_val, y_val)
    custom_focal_loss = focal_loss(gamma=2., alpha=4.)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=custom_focal_loss,
                  #loss=CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')
    callbacks_list = [model_checkpoint, early_stopping]

    # Training the model
    history = model.fit(x_train, y_train_onehot,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val_onehot),
                        callbacks=callbacks_list)
    return history

def train_model_vgg_with_augmentation_focal(model, x_train, y_train, x_val, y_val, learning_rate=0.001,  epochs=50, batch_size=128, patience=10, model_path='B/vgg_with_aug_focal_loss.h5'):
    x_train, y_train, x_val, y_val, y_train_onehot, y_val_onehot = preprocess_data(x_train, y_train, x_val, y_val)
    custom_focal_loss = focal_loss(gamma=2., alpha=4.)

    # Define the data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Create a training data generator
    train_generator = train_datagen.flow(x_train, y_train_onehot, batch_size=batch_size)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  #loss=CategoricalCrossentropy(from_logits=False),
                  loss=custom_focal_loss,
                  metrics=['accuracy'])
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')
    callbacks_list = [model_checkpoint, early_stopping]
    # Training the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(x_train) // batch_size,  # Number of steps per epoch
        validation_data=(x_val, y_val_onehot),
        callbacks=callbacks_list
    )
    return history


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


def evaluate_and_visualize_model(model_path, x_test, y_test):
    x_test = x_test.astype('float32') / 255
    y_test_one_hot = to_categorical(y_test, num_classes=9)  # One-hot encode y_test for model evaluation
    focal_loss_fixed = focal_loss()
    model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(x_test, y_test_one_hot)
    print(f'Test Loss: {loss:.2f}, Test Accuracy: {accuracy * 100:.2f}%')

    # Make predictions
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test  # Using non-one-hot encoded y_test for classification report

    # Compute and plot the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    # Calculate precision and recall
    precision = precision_score(true_classes, predicted_classes, average='macro')
    recall = recall_score(true_classes, predicted_classes, average='macro')

    # Print accuracy, precision, and recall
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Print classification report
    print("\nClassification Report:\n")
    print(classification_report(true_classes, predicted_classes, target_names=class_label_names.values()))



