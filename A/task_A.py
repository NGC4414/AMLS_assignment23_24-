import sys
from A.train_and_test_A import analyze_and_visualize_data, load_data_pneumonia, pneumoniaCNNPredict, pneumoniaLogRegrPredict

def print_menu_A():
    print("Select models for Task A:")
    print("1. Logistic Regression")
    print("2. Convolutional Neural Network (CNN)")
    print("3. Exit program")
    #print("3. Support Vector Machines (SVM)")

# def select_models_A():

#     models_A = []
#     while True:
#         try:
#             choices = [int(x) for x in input("Enter model numbers for Task A (e.g., 1 2 3): ").split()]
#             if all(choice in [1, 2, 3] for choice in choices):
#                 models_A.extend(choices)
#                 break
#             else:
#                 print("Invalid choice. Please enter valid model numbers.")
#         except ValueError:
#             print("Invalid input. Please enter numbers.")

#     return models_A

def train_and_test_A(models, x_train, y_train, x_val, y_val, x_test, y_test):
    for model_choice in models:
        if model_choice == 1:
            pneumoniaLogRegrPredict(x_train, y_train, x_val, y_val, x_test, y_test)
        elif model_choice == 2:
            ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()
            analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test, x_train, n_channels=1, length=20)
            pneumoniaCNNPredict(x_test, y_test)
        #elif model_choice == 3:
            # SVMModel(train_data, val_data, test_data)
            #print('SVM')



# def run_task_A():            # function that runs all task A
#     ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()

#     while True:
#         print_menu_A()

#         # Get user input for the option
#         try:
#             option = int(input("Enter the option number: "))
#         except ValueError:
#             print("Invalid input. Please enter a number.")
#             continue

#         if option == 1:
#             models = select_models_A()
#             train_and_test_A(models, x_train, y_train, x_val, y_val, x_test, y_test)

#         elif option == 2:
#             models = select_models_A()
#             train_and_test_A(models, x_train, y_train, x_val, y_val, x_test, y_test)

#         elif option == 3:
#             sys.exit()

#             #models = select_models_A()
#             #train_and_test_A(models, x_train, y_train, x_val, y_val, x_test, y_test)
        
#         #elif option == 4:
#         #    sys.exit()

#         else:
#             print("Invalid option. Please enter 1, 2 or 3")


def run_task_A():            # function that runs all task A
    ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()

    while True:
        print_menu_A()

        # Get user input for the option
        try:
            option = int(input("Enter the option number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if option == 1:
            pneumoniaLogRegrPredict(x_train, y_train, x_val, y_val, x_test, y_test)
        elif option == 2:
            pneumoniaCNNPredict(x_test, y_test)
            ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()
            analyze_and_visualize_data(x_train, y_train, x_val, y_val, x_test, y_test, x_train, n_channels=1, length=20)
            pneumoniaCNNPredict(x_test, y_test)
        elif option == 3:
            sys.exit()
        else:
            print("Invalid option. Please enter 1, 2, or 3.")
