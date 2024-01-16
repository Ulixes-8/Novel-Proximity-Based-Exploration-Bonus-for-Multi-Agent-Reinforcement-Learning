import optuna
from train import train
from evaluation import evaluation
from copy import deepcopy
import hyperparameters

def objective(trial):
    # Suggest hyperparameters
    suggested_hyperparams = {
        'initial_decay_factor': trial.suggest_float('initial_decay_factor', .1,10),
        'decay_rate': trial.suggest_float('decay_rate', 0.01, 0.99),
        'scaling_factor': trial.suggest_float('scaling_factor', 0.0001, 1),
        'probability': trial.suggest_float('probability', 0.01, 0.99),
    }

    # Temporarily override hyperparameters
    original_hyperparams = deepcopy(hyperparameters.eb_marl_hyperparameters)
    hyperparameters.eb_marl_hyperparameters.update(suggested_hyperparams)

    # Train and evaluate agents
    print("Training with hyperparameters: ", hyperparameters.eb_marl_hyperparameters)
    reward = evaluation(hyperparameters.agent_hyperparameters['agent_choice'])

    # Restore original hyperparameters
    hyperparameters.eb_marl_hyperparameters = original_hyperparams

    return reward

def main():
    study = optuna.create_study(direction='maximize', study_name='my_study_2', storage='sqlite:///db.sqlite3', load_if_exists=True)
    study.optimize(objective, n_trials=1000)  # Adjust n_trials as needed

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best mean reward: {study.best_value}")

if __name__ == "__main__":
    main()
