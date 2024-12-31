This software was written as a final year thesis. It received a final mark of 81%, ranking 8th out of 412 computer science theses at the University of Birminghmam. 

# Project Repository

## Running the Experiment Pipeline

To run the experiment pipeline, which includes all experiments performed in this thesis, as well as others, follow these steps:

1. **Clone the repo.**
2. **Download the dependencies** using the `requirements.txt` file.
3. **Navigate to the `hyperparameters.py` file** and ensure that the hyperparameters are set to your liking. If your goal is to replicate the experiments in this thesis, please pay special attention to `eb_marl_multiple_parameters`, which contain the hyperparameters for the PB algorithm.
4. **Run the `main.py` file.**
5. **Read the menu** and select the experiment you want to run.
6. **Follow the instructions** on the screen. You will be presented with a host of confirmations as well as the adjacency matrices for the agents in the experiment. If you plan to run multiple experiments simultaneously, open separate terminals for them. Please ensure you have a powerful CPU with several cores, sufficient RAM (32GB+), and a very large swap space (256GB+) to avoid the OOM killer after several hours of training.

## Key Files for Review

If you are grading my paper, then the files `observer.py`, `twelve_experiments.py`, `eb_marl_agent.py` may be of most interest to you.

## Guidelines for Further Experimentation

If you are building upon this repo or attempting to use it for your own experiments, please take note of the following guidelines:

1. **Editing State Space and Configurations:** Make necessary changes in `pettingZoo/PettingZoo/pettingzoo/mpe/simple_spread/simple_spread.py`.
2. **Defining New Experiments:** Edit or define new experiments in the `hyperparameters.py` file. Make sure they are added to `experiments_choice` array.
3. **Tailoring the Experiment Pipeline:** The experiment pipeline is detailed in `twelve_experiments.py`. Adjust this file to your specific use cases.
4. **Training and Evaluation:** The pipeline trains and evaluates agents based on inputs to `hyperparameters.py`. Implement any specific functionality as needed.
5. **Algorithm Logic:** Logic for UCB and PB agents is contained in `ucb_marl_agent.py` and `eb_marl_agent.py` respectively. 
6. **Observer/Oracle Class:** Located in `observer.py`, for adding new metrics compute from the Universal N-table.
7. **Experimentation Needs:** Primarily, you will only need to edit a select few files for most experimentation purposes.
8. **Developing a New Algorithm:** If developing a new algorithm, create a new file and ensure compatibility with the existing codebase.
9. **Network Topologies:** Define all new network topologies in `hyperparameters.py`.

