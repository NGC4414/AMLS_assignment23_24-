import os
import sys
from B.train_and_test_B import (create_efficientnetb0, evaluate_and_visualize_efficientnet, image_transformation, plot_training_history_efficientnet,  
                                create_tinyvgg, evaluate_and_visualize_model, 
                                load_data_pathmnist, plot_class_distribution_pathmnist, preprocess_data_eff, 
                                show_sample_images, train_efficientnetb0, train_model, plot_training_history)


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

def print_menu_B():
    print("Select models for Task B:")
    print("1. Train Convolutional Neural Network (TinyVGG)")
    print("2. Use Pre-Trained Convolutional Neural Network (TinyVGG)")
    print("3. Train EfficientNetB0 Model")  
    print("4. Use Pre-Trained EfficientNetB0 Model") 
    print("5. Exit program")



def run_task_B():                               
    ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pathmnist()

    while True:
        print_menu_B()
        try:
            option = int(input("Enter the option number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if option == 1:
            # Train a new model
            input_shape = (28, 28, 3)
            n_classes = 9
            plot_class_distribution_pathmnist(y_train, y_val, y_test)
            show_sample_images(x_train, y_train, class_label_names)
            model = create_tinyvgg(input_shape, hidden_units=10, output_shape=n_classes)
            model.summary()
            history = train_model(model, x_train, y_train, x_val, y_val, learning_rate=0.001, momentum=0.9, epochs=50, batch_size=128, patience=5, model_path='./tinyvgg.h5')
            plot_training_history(history)
            evaluate_and_visualize_model('./tinyvgg.h5', x_test, y_test)

        elif option == 2:
            plot_class_distribution_pathmnist(y_train, y_val, y_test)
            show_sample_images(x_train, y_train, class_label_names)
            # Use pre-trained model
            if os.path.exists('./tinyvgg.h5'):
                evaluate_and_visualize_model('./tinyvgg.h5', x_test, y_test)
            else:
                print("Pre-trained model not found. Please train a model first.")

        elif option == 3:
            # Train a new EfficientNetB0 model
            n_classes = 9
            plot_class_distribution_pathmnist(y_train, y_val, y_test)
            show_sample_images(x_train, y_train, class_label_names)
            model = create_efficientnetb0(n_classes, l2_reg=0.001)

            # Preprocess data for EfficientNetB0
            x_train_norm, y_train_enc, x_val_norm, y_val_enc = preprocess_data_eff(x_train, y_train, x_val, y_val)

            # Create train_generator
            train_generator = image_transformation(x_train_norm, y_train_enc, num_classes=9, batch_size=32)

            # Train EfficientNetB0
            history = train_efficientnetb0(model, train_generator, x_val_norm, y_val_enc, learning_rate=0.001, epochs=50, batch_size=32, patience=5, model_path='./efficientnetb0_model.h5')
            
            plot_training_history_efficientnet(history)
            evaluate_and_visualize_efficientnet('./efficientnetb0_model.h5', x_test, y_test)
            
        elif option == 4:
            # Use pre-trained EfficientNetB0 model
            if os.path.exists('./efficientnetb0_model.h5'):
                evaluate_and_visualize_efficientnet('./efficientnetb0_model.h5', x_test, y_test)
            else:
                print("Pre-trained EfficientNetB0 model not found. Please train a model first.")

        elif option == 5:
            sys.exit()

        else:
            print("Invalid option. Please enter 1, 2, 3, 4, or 5")        
