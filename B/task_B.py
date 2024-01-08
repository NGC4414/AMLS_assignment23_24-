import os
import sys
from B.train_and_test_B import ( create_vgg, evaluate_and_visualize_model, 
                                load_data_pathmnist, plot_class_distribution_pathmnist, 
                                show_sample_images, train_model, plot_training_history, train_model_with_augmentation)


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
    print("1. Train TinyVGG without Data Augmentation")
    print("2. Train TinyVGG with Data Augmentation")
    print("3. Use Pre-Trained TinyVGG without Data Augmentation")
    print("4. Use Pre-Trained TinyVGG with Data Augmentation")
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

        if option in [1, 2]:
            input_shape = (28, 28, 3)
            n_classes = 9
            plot_class_distribution_pathmnist(y_train, y_val, y_test)
            show_sample_images(x_train, y_train, class_label_names)
            model = create_vgg(input_shape, hidden_units=10, output_shape=n_classes)
            model.summary()

            if option == 1:
                # Train without data augmentation
                history = train_model(model, x_train, y_train, x_val, y_val, learning_rate=0.001, momentum=0.9, epochs=50, batch_size=128, patience=5, model_path='./tinyvgg_no_aug.h5')
            else:
                # Train with data augmentation
                history = train_model_with_augmentation(model, x_train, y_train, x_val, y_val, learning_rate=0.001, momentum=0.9, epochs=50, batch_size=128, patience=5, model_path='./tinyvgg_with_aug.h5')

            plot_training_history(history)
            if option == 1:
                evaluate_and_visualize_model('./tinyvgg_no_aug.h5', x_test, y_test)
            else:
                evaluate_and_visualize_model('./tinyvgg_with_aug.h5', x_test, y_test)

        elif option == 3:
            # Use pre-trained TinyVGG without data augmentation
            if os.path.exists('./tinyvgg_no_aug.h5'):
                plot_class_distribution_pathmnist(y_train, y_val, y_test)
                show_sample_images(x_train, y_train, class_label_names)
                evaluate_and_visualize_model('./tinyvgg_no_aug.h5', x_test, y_test)
            else:
                print("Pre-trained model without data augmentation not found. Please train a model first.")

        elif option == 4:
            # Use pre-trained TinyVGG with data augmentation
            if os.path.exists('./tinyvgg_with_aug.h5'):
                plot_class_distribution_pathmnist(y_train, y_val, y_test)
                show_sample_images(x_train, y_train, class_label_names)
                evaluate_and_visualize_model('./tinyvgg_with_aug.h5', x_test, y_test)
            else:
                print("Pre-trained model with data augmentation not found. Please train a model first.")

        elif option == 5:
            sys.exit()

        else:
            print("Invalid option. Please enter a valid number.")        



