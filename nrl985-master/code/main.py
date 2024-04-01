from hyperparameters import experiments_choice
import time
from twelve_experiments import experiment_pipeline

## When you define new experiments and add them to the experiments_choice array, they will automatically show up here. No need to do anything. 

def menu():
    """The menu which runs the experiment pipeline"""
    print('Please select which experiment you would like to run by entering the corresponding number (e.g., "1" for experiment 1.).')
    print('There are 17 experiments total, so enter a number between 1 and 17.')
    print('Please ensure that the hyperparameters for both the algorithm and the environment are set to their desired values in hyperparameters.py before running an experiment.')
    print('Please also note that the code is parallelized and so will be utilizing a lot of your CPU.\n')
    started = False
    for i, experiment_group in enumerate(experiments_choice, start=1):
        experiment_name = "UCB vs. "
        experiment_name += experiment_group[0]['experiment_name']
        print(f"{i}: {experiment_name}")

    try:
        choice = int(input("Enter your choice: "))
    except ValueError:
        print("Please enter a valid integer.")
        return

    if 1 <= choice <= len(experiments_choice):
        print(f"Experiment {choice} selected.\n")
        started = True
        print('Time started')
        start_time = time.time()

        experiment_pipeline(experiments_choice[choice-1])
        print("Experiment successful.")
    else:
        print("Invalid choice. Please select a number between 1 and 17.")

    if started: 
        print('\nTime finished.')
        print(f'--- {time.time() - start_time} seconds ---')
        print('\nHave a good day')



# menu()
if __name__ == '__main__':
    menu()