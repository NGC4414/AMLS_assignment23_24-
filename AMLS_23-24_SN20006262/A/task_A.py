import sys
import os
from A.train_and_test_A import (analyze_and_visualize_data, load_data_pneumonia, plot_roc_curves, plot_roc_curves_pca, 
                                plot_training_history, pneumoniaCNNPredict, 
                                pneumoniaLogRegrPredict, train_and_evaluate_svms, train_and_evaluate_svms_pca, train_and_save_cnn)

def print_menu_A():
    print("Select models for Task A:")
    print("1. Logistic Regression")
    print("2. SVM without PCA")
    print("3. SVM with PCA")
    print("4. Train Convolutional Neural Network (CNN)")
    print("5. Use Pre-Trained Convolutional Neural Network (CNN)")
    print("6. Exit program")

def run_task_A(): 
    #load the data
    ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()

    while True:
        print_menu_A()
        try:
            option = int(input("Enter the option number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if option == 1:
            # Train new Logistic Regression model
            analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test)
            pneumoniaLogRegrPredict(x_train, y_train, x_val, y_val, x_test, y_test)
        elif option == 2:
            # Train new SVM models without PCA
            analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test)
            models, scalers = train_and_evaluate_svms(x_train, y_train, x_val, y_val, x_test, y_test,
                                          C_linear=1.0, C_rbf=2, gamma_rbf='scale', 
                                          C_poly=1.0, degree_poly=4, 
                                          C_sigmoid=0.5, tol=1e-3, max_iter=-1)
            
            plot_roc_curves(models, scalers, x_test, y_test)
        elif option == 3:
            # Train new SVM models with PCA
            analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test)
            models, scalers, pcas = train_and_evaluate_svms_pca(x_train, y_train, x_val, y_val, x_test, y_test, n_components=0.95,
                                                    C_linear=1.0, C_rbf=2.0, gamma_rbf='scale', 
                                                    C_poly=1.0, degree_poly=4, 
                                                    C_sigmoid=0.5, tol=1e-3, max_iter=-1)

            plot_roc_curves_pca(models, scalers, pcas, x_test, y_test)
        elif option == 4:
            # Train new CNN model
            analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test)
            trained_model, history = train_and_save_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
            plot_training_history(history)
            pneumoniaCNNPredict(x_test, y_test)
        elif option == 5:
            # Use pre-trained CNN model
            if os.path.exists('A/pretrained_model.h5'):
                analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test)
                pneumoniaCNNPredict(x_test, y_test)
            else:
                print("Pre-trained model not found. Please train a model first.")

        elif option == 6:
            sys.exit()
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, 5 or 6.")
