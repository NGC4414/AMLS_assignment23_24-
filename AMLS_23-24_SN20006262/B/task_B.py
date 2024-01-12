import os
import sys
from B.train_and_test_B import (create_vgg, evaluate_and_visualize_model, 
                                load_data_pathmnist, plot_class_distribution_pathmnist, 
                                show_sample_images, plot_training_history, train_model_vgg_cross, 
                                train_model_vgg_focal, train_model_vgg_with_augmentation_cross, train_model_vgg_with_augmentation_focal)


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
    
    print("1. Train simplified VGG without Data Augmentation (focal loss)")
    print("2. Train simplified VGG with Data Augmentation (focal loss)")
    
    print("3. Train simplified VGG without Data Augmentation (categorical cross entropy)")
    print("4. Train simplified VGG with Data Augmentation (categorical cross entropy)")
    
    print("5. Use Pre-Trained simplified VGG without Data Augmentation (focal loss)")
    print("6. Use Pre-Trained simplified VGG with Data Augmentation (focal loss)")

    print("7. Use Pre-Trained simplified VGG without Data Augmentation (categorical cross entropy)")
    print("8. Use Pre-Trained simplified VGG with Data Augmentation (categorical cross entropy)")

    print("9. Exit program")

def run_task_B():                               
    ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pathmnist()

    while True:
        print_menu_B()
        try:
            option = int(input("Enter the option number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if option in [1, 2, 3, 4]:
            input_shape = (28, 28, 3)
            n_classes = 9
            plot_class_distribution_pathmnist(y_train, y_val, y_test)
            show_sample_images(x_train, y_train, class_label_names)
            model = create_vgg(input_shape, hidden_units=10, output_shape=n_classes)
            model.summary()

            if option == 1:
                # Train without data augmentation using focal loss
                history = train_model_vgg_focal(model, x_train, y_train, x_val, y_val, learning_rate=0.001, epochs=50, batch_size=128, patience=5, model_path='B/vgg_no_aug_focal_loss.h5')
                plot_training_history(history)
                evaluate_and_visualize_model('B/vgg_no_aug_focal_loss.h5', x_test, y_test)

            elif option == 2:
                # Train with data augmentation using  focal loss
                history = train_model_vgg_with_augmentation_focal(model, x_train, y_train, x_val, y_val, learning_rate=0.001, epochs=50, batch_size=128, patience=5, model_path='B/vgg_with_aug_focal_loss.h5')
                plot_training_history(history)
                evaluate_and_visualize_model('B/vgg_with_aug_focal_loss.h5', x_test, y_test)

            elif option == 3:
                # Train without data augmentation using categorical cross entropy
                history = train_model_vgg_cross(model, x_train, y_train, x_val, y_val, learning_rate=0.001, epochs=50, batch_size=128, patience=5, model_path='B/vgg_no_aug_crossentropy.h5')
                plot_training_history(history)
                evaluate_and_visualize_model('B/vgg_no_aug_crossentropy.h5', x_test, y_test)

            elif option == 4:
                # Train with data augmentation using categorical cross entropy
                history = train_model_vgg_with_augmentation_cross(model, x_train, y_train, x_val, y_val, learning_rate=0.001, epochs=50, batch_size=128, patience=5, model_path='B/vgg_with_aug_crossentropy.h5')
                plot_training_history(history)
                evaluate_and_visualize_model('B/vgg_with_aug_crossentropy.h5', x_test, y_test)

        if option in [5, 6, 7, 8]:
            input_shape = (28, 28, 3)
            n_classes = 9
            plot_class_distribution_pathmnist(y_train, y_val, y_test)
            show_sample_images(x_train, y_train, class_label_names)
            # model = create_vgg(input_shape, hidden_units=10, output_shape=n_classes)
            # model.summary()

            if option == 5:
                # Use pre-trained VGG without data augmentation (focal loss)
                if os.path.exists('B/vgg_no_aug_focal_loss.h5'):
                    evaluate_and_visualize_model('B/vgg_no_aug_focal_loss.h5', x_test, y_test)
                else: 
                    print("Pre-trained model with data augmentation not found. Please train a model first.")

            elif option == 6:
                # Use pre-trained VGG without data augmentation (focal loss)
                if os.path.exists('B/vgg_with_aug_focal_loss.h5'):
                    evaluate_and_visualize_model('B/vgg_with_aug_focal_loss.h5', x_test, y_test)
                else: 
                    print("Pre-trained model with data augmentation not found. Please train a model first.")

            elif option == 7:
                #Use pre-trained VGG without data augmentation (categorical crossentropy)
                if os.path.exists('B/vgg_no_aug_crossentropy.h5'):
                    evaluate_and_visualize_model('B/vgg_no_aug_crossentropy.h5', x_test, y_test)
                else: 
                    print("Pre-trained model with data augmentation not found. Please train a model first.")
            
            elif option == 8:
                if os.path.exists('B/vgg_with_aug_crossentropy.h5'):
                    evaluate_and_visualize_model('B/vgg_with_aug_crossentropy.h5', x_test, y_test)
                #Use pre-trained VGG with data augmentation (categoricla crossentropy)
                else: 
                    print("Pre-trained model with data augmentation not found. Please train a model first.")

        elif option == 9:
            sys.exit()

        else:
            print("Invalid option. Please enter a valid number.")



