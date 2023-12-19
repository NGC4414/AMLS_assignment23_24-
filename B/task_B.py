from B.train_and_test_B import load_data_pathmnist


def print_menu_B():
    print("Select models for Task B:")
    print("1. Multivariate logistic regression")
    

def select_models_B():

    models_B = []
    while True:
        try:
            choices = [int(x) for x in input("Enter model numbers for Task B (e.g., 1 2): ").split()]
            if all(choice in [1, 2] for choice in choices):
                models_B.extend(choices)
                break
            else:
                print("Invalid choice. Please enter valid model numbers.")
        except ValueError:
            print("Invalid input. Please enter numbers.")

    return models_B

def run_task_B():
   # ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = load_data_pneumonia()

    while True:
        print_menu_B()

        # Get user input for the option
        try:
            option = int(input("Enter the option number (1 or 2): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if option == 1:
            models = select_models_B()
            train_and_test_B(models, x_train, y_train, x_val, y_val, x_test, y_test)
            
        elif option == 2:
            pass
        else:
            print("Invalid option. Please enter 1 or 2.")


