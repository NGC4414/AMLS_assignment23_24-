import sys
from A.train_and_test_A import analyze_and_visualize_data, load_data_pneumonia, pneumoniaCNNPredict, pneumoniaLogRegrPredict

def print_menu_A():
    print("Select models for Task A:")
    print("1. Logistic Regression")
    print("2. Convolutional Neural Network (CNN)")
    print("3. Exit program")



def run_task_A():            # function that runs all task A
    ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()

    while True:
        print_menu_A()

        # Get user input for the option
        try:
            option = int(input("Enter the option number (1, 2 or 3): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if option == 1:
            pneumoniaLogRegrPredict(x_train, y_train, x_val, y_val, x_test, y_test)
        elif option == 2:
            ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()
            analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test, x_train, n_channels=1, length=20)
            pneumoniaCNNPredict(x_test, y_test)
        elif option == 3:
            sys.exit()
        else:
            print("Invalid option. Please enter 1, 2, or 3.")
