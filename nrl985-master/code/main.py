"""This will start the program.  Contains a small menu 
    which can either train a model or run a trained model"""
from file_management import show_all_files
from train import train
from evaluation import evaluation
from show import play_on_show
from hyperparameters import agent_hyperparameters
import time

def menu():




    """The menu which runs the main program"""

    print('Please choose an option:\n'
    '1. Train Agents (Please check hyperparameters and right Agent chosen)\n'
    '2. Test Agents (Please check hyperparameters used)\n'
    '3. Load and Show Agents')
    choice = int(input())
    print('Time started')
    start_time = time.time()
    
    if choice == 1:
        print('\n')
        print('Episode Number ran')
        train(agent_hyperparameters['agent_choice'])

    elif choice == 2:
        print('\n')
        evaluation(agent_hyperparameters['agent_choice'])

    elif choice == 3:
        print('\n')
        _play_agents()

    print('\n')
    print(f'--- {time.time() - start_time} seconds ---')
    print('\n')
    print('Have a good day')


def _play_agents():

    """This will play a certain agent which is saved on the system"""

    agent_files = show_all_files()
    print('Please choose an agent file to use')
    for i, agent_file in enumerate(agent_files):
        print(f'{i}. {agent_file}')
    choice = int(input())
    print('\n')
    filename = agent_files[choice] 
    play_on_show(filename)


# menu()
if __name__ == '__main__':
    menu()