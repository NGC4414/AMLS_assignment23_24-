from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, ReLU, Dropout
import seaborn as sns
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
from keras.regularizers import l2


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

# Define a dictionary that maps class labels to their names
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

# Function to print class label descriptions - this function is called in plot_class_distribution_pathmnist(y_train, y_val, y_test)
# to provide more information about the names of each class
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

############################################VGG#############################################################################################

def preprocess_data(x_train, y_train, x_val, y_val):
    # Normalize images
    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=9)
    y_val = to_categorical(y_val, num_classes=9)

    return x_train, y_train, x_val, y_val, 

    
def create_tinyvgg(input_shape, hidden_units, output_shape):
    model = Sequential()
    
    # Conv Block 1
    model.add(Conv2D(hidden_units, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
    model.add(ReLU())
    model.add(Conv2D(hidden_units, kernel_size=3, strides=1, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))
    
    # Conv Block 2
    model.add(Conv2D(hidden_units, kernel_size=3, strides=1, padding='same'))
    model.add(ReLU())
    model.add(Conv2D(hidden_units, kernel_size=3, strides=1, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Classifier
    model.add(Flatten())
    model.add(Dense(output_shape, activation='softmax'))

    return model



def train_model(model, x_train, y_train, x_val, y_val, learning_rate=0.001, momentum=0.9, epochs=50, batch_size=128, patience=5, model_path='./tinyvgg.h5'):
    x_train, y_train, x_val, y_val = preprocess_data(x_train, y_train, x_val, y_val)
    # Compile the model
    model.compile(optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
                  loss=CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')
    callbacks_list = [model_checkpoint, early_stopping]

    # Training the model
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks_list)
    
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
    y_test = to_categorical(y_test, num_classes=9)
    model = load_model(model_path)

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.2f}, Test Accuracy: {accuracy * 100:.2f}%')

    # Make predictions
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Compute and plot the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()
################################################################################################################################################
def preprocess_data_eff(x_train, y_train, x_val, y_val, target_size=(224, 224)):
    x_train = tf.convert_to_tensor(x_train)
    x_val = tf.convert_to_tensor(x_val)
    
    # Resize images
    x_train_resized = tf.image.resize(x_train, target_size)
    x_val_resized = tf.image.resize(x_val, target_size)

    # Normalize images
    x_train_normalized = x_train_resized / 255.0
    x_val_normalized = x_val_resized / 255.0

    # Convert labels to one-hot encoding
    y_train_encoded = to_categorical(y_train, num_classes=9)
    y_val_encoded = to_categorical(y_val, num_classes=9)

    return x_train_normalized, y_train_encoded, x_val_normalized, y_val_encoded

def image_transformation(x_train, y_train, num_classes=9, batch_size=128):
    # Define the ImageDataGenerator with augmentation for training data
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    return train_generator

def create_efficientnetb0(n_classes, l2_reg=0.001):
    # Load EfficientNetB0 with pre-trained ImageNet weights
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Add custom layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)  # Dropout for regularization
    output = Dense(n_classes, activation='softmax', kernel_regularizer=l2(l2_reg))(x)  # Final Dense layer for classification

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    return model


def train_efficientnetb0(model, train_generator, x_val, y_val, learning_rate=0.001, epochs=5, batch_size=32, patience=5, model_path='./efficientnetb0_model.h5'):
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CategoricalCrossentropy(), metrics=['accuracy'])

    # Callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')

    # Training the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

    return history


def plot_training_history_efficientnet(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()

def evaluate_and_visualize_efficientnet(model_path, x_test, y_test):
    # Resize test images to match the input shape expected by the model
    x_test_resized = tf.image.resize(x_test, (224, 224))
    x_test_resized = x_test_resized / 255.0
    y_test_encoded = to_categorical(y_test, num_classes=9)

    # Load the trained model
    model = load_model(model_path)

    # Evaluate the model on resized test data
    loss, accuracy = model.evaluate(x_test_resized, y_test_encoded)
    print(f'Test Loss: {loss:.2f}, Test Accuracy: {accuracy * 100:.2f}%')

    # Make predictions
    predictions = model.predict(x_test_resized)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_encoded, axis=1)

    # Compute and plot the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()