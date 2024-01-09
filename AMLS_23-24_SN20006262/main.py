import sys
from A.task_A import run_task_A
from B.task_B import run_task_B

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <task>")
        sys.exit(1)

    task = sys.argv[1]

    if task == 'A':
        run_task_A() #run_task_A() is a function inside the task_A file in the A folder 
    elif task == 'B':
        run_task_B() #run_task_B() is a function inside the task_B file in the B folder
    else:
        print("Invalid task. Please choose 'A' or 'B'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
