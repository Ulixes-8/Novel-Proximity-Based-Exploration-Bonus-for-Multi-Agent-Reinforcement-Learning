import dill
import matplotlib.pyplot as plt
from utils import _pickle_loader, get_average_values, _open, create_array

# The current values being used.
# c: 0.02,
# probability: 0.1,
# 'gamma': 0.8,
# 'greedy': 0.5,
# 'alpha': 0.8

def main():

    """
    Starts the program
    """

    print('Please decide what you need to create:')
    print('1. Data from 4 agents\n'
          '2. Basic Graph Data from 12 agents\n'
          '3. Star Graph Data from 12 agents\n'
          '4. Ring Graph Data from 12 agents\n'
          '5. All data from 12 agents\n'
          '6. Best Plots for the 12 agents\n'
          '7. Star 12 \n'
          '8. Dynamic Check\n')
    choice = int(input())
    
    print('How often should it output a test episode:')
    test_episode = int(input())
    if choice == 1:
        main_four(test_episode)
    elif choice == 2:
        create_basic(test_episode)
    elif choice == 3: 
        create_star(test_episode)
    elif choice == 4:
        create_ring(test_episode)
    elif choice == 5:
        main_twelve(test_episode)
    elif choice == 6:
        best_plots(test_episode)
    elif choice == 7:
        create_12_star(test_episode)
    elif choice == 8:
        create_dynamic_check(test_episode)        
    else:
        print('Not an option')


def create_12_star(episode_output):

    """
    Creates the Star graph for when we split out the agents 7 and 8
    episode_output - How often the data should be graphed
    """

    FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_10000_0_'
    RANDOM_PART = 'saved_data_twelve/AgentType.EB_Lidard_12_10_1000_0_'
    # RANDOM_PART = 'saved_data_twelve/AgentType.RANDOM_12_10_10000_0_'

    filename = FIRST_PART+'22.pkl'
    episodes = [file for file in _pickle_loader(filename)][0]['reward'][2][::episode_output] 


    random_switch = [0]
    line_gamma_10 = [2]
    fully_connected = [3]
    star_zero_two = [10]
    star = [22]

    get_average_line_gamma_10 = get_average_values(line_gamma_10, FIRST_PART)[::episode_output]
    get_average_fully_connected = get_average_values(fully_connected, FIRST_PART)[::episode_output]
    get_average_star_zero_two = get_average_values(star_zero_two, FIRST_PART)[::episode_output]
    get_average_random_switch = get_average_values(random_switch, RANDOM_PART)[::episode_output]
    star_data = get_average_values(star, FIRST_PART)[::episode_output]

    plt.figure()
    plt.plot(episodes, get_average_random_switch[:, 3], label = f'Random')
    plt.plot(episodes, get_average_fully_connected[:,3], label='Fully Connected')
    plt.plot(episodes, get_average_star_zero_two[:,3], label='Star Gamma 2')
    plt.plot(episodes, get_average_line_gamma_10[:,3], label='Line Gamma 10')
    plt.plot(episodes, star_data[:, 3], label = f'Star 7')
    plt.plot(episodes, star_data[:, 6], label = f'Agent 7 - Centre')
    plt.plot(episodes, star_data[:,7], label = f'Agent 8')

    plt.title(f'Evaluation of Star when splitting Agents 7 & 8.')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs_twelve/evaluation_switch_star_{episode_output}.png')


def create_dynamic_check(episode_output):

    """
    Creates the Dynamic graph for when we split out the agents 7 and 8
    episode_output - How often the data should be graphed
    """

    FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_10000_0_'
    RANDOM_PART = 'saved_data_twelve/AgentType.EB_Lidard_12_10_1000_0_'
    # RANDOM_PART = 'saved_data_twelve/AgentType.RANDOM_12_10_10000_0_'

    filename = FIRST_PART+'22.pkl'
    episodes = [file for file in _pickle_loader(filename)][0]['reward'][2][::episode_output] 


    random_switch = [0]
    line_gamma_10 = [2]
    fully_connected = [3]
    star_zero_two = [10]
    dynamic = [23]

    get_average_line_gamma_10 = get_average_values(line_gamma_10, FIRST_PART)[::episode_output]
    get_average_fully_connected = get_average_values(fully_connected, FIRST_PART)[::episode_output]
    get_average_star_zero_two = get_average_values(star_zero_two, FIRST_PART)[::episode_output]
    get_average_random_switch = get_average_values(random_switch, RANDOM_PART)[::episode_output]
    star_data = get_average_values(dynamic, FIRST_PART)[::episode_output]

    plt.figure()
    plt.plot(episodes, get_average_random_switch[:, 3], label = f'Random')
    plt.plot(episodes, get_average_fully_connected[:,3], label='Fully Connected')
    plt.plot(episodes, get_average_star_zero_two[:,3], label='Star Gamma 2')
    plt.plot(episodes, get_average_line_gamma_10[:,3], label='Line Gamma 10')
    plt.plot(episodes, star_data[:, 3], label = f'Dynamic')
    plt.plot(episodes, star_data[:, 6], label = f'Agent 7')
    plt.plot(episodes, star_data[:,7], label = f'Agent 8')

    plt.title(f'Evaluation of Dynamic when splitting Agents 7 & 8.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs_twelve/evaluation_switch_dynamic_{episode_output}.png')



def best_plots(episode_output):
    """
    Creates the graphs for the best network topolgies (EG where the agents are just not random) as worked out by main_analysis.py and episodes == 100.
    episode_output - How often the data should be graphed
    """

    FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_10000_0_'
    # RANDOM_PART = 'saved_data_twelve/AgentType.RANDOM_12_10_10000_0_'
    RANDOM_PART = 'saved_data_twelve/AgentType.EB_Lidard_12_10_1000_0_'
    

    filename = FIRST_PART+'0.pkl'
    episodes = [file for file in _pickle_loader(filename)][0]['reward'][2][::episode_output]

    random_switch = [0]
    dynamic = [0]
    line_gamma_10 = [2]
    fully_connected = [3]
    ring_two = [4]
    ring_four = [5]
    ring_six = [6]
    ring_eight = [7]
    ring_ten = [8]
    star_zero = [9]
    star_zero_two = [10]
    star_one = [11]
    star_two = [12]
    star_three = [13]
    star_four = [14]
    star_five = [15]
    star_six = [16]
    star_seven = [17]
    star_eight = [18]
    star_nine = [19]
    star_ten = [20]
    star_eleven = [21]

    get_average_random_switch = get_average_values(random_switch, RANDOM_PART)[::episode_output]
    get_average_dynamic = get_average_values(dynamic, FIRST_PART)[::episode_output]
    get_average_line_gamma_10 = get_average_values(line_gamma_10, FIRST_PART)[::episode_output]
    get_average_fully_connected = get_average_values(fully_connected, FIRST_PART)[::episode_output]
    get_average_ring_two = get_average_values(ring_two, FIRST_PART)[::episode_output]
    get_average_ring_four = get_average_values(ring_four, FIRST_PART)[::episode_output]
    get_average_ring_six = get_average_values(ring_six, FIRST_PART)[::episode_output]
    get_average_ring_eight = get_average_values(ring_eight, FIRST_PART)[::episode_output]
    get_average_ring_ten = get_average_values(ring_ten, FIRST_PART)[::episode_output]
    get_average_star_zero = get_average_values(star_zero, FIRST_PART)[::episode_output]
    get_average_star_one = get_average_values(star_one, FIRST_PART)[::episode_output]
    get_average_star_two = get_average_values(star_two, FIRST_PART)[::episode_output]
    get_average_star_three = get_average_values(star_three, FIRST_PART)[::episode_output]
    get_average_star_four = get_average_values(star_four, FIRST_PART)[::episode_output]
    get_average_star_five = get_average_values(star_five, FIRST_PART)[::episode_output]
    get_average_star_six = get_average_values(star_six, FIRST_PART)[::episode_output]
    get_average_star_seven = get_average_values(star_seven, FIRST_PART)[::episode_output]
    get_average_star_eight = get_average_values(star_eight, FIRST_PART)[::episode_output]
    get_average_star_nine = get_average_values(star_nine, FIRST_PART)[::episode_output]
    get_average_star_ten = get_average_values(star_ten, FIRST_PART)[::episode_output]
    get_average_star_eleven = get_average_values(star_eleven, FIRST_PART)[::episode_output]
    get_average_star_zero_two = get_average_values(star_zero_two, FIRST_PART)[::episode_output]

    # i = 0 (Agents 0 and 6)
    plt.figure()
    plt.plot(episodes, get_average_random_switch[:, 0], label = f'Random')
    plt.plot(episodes, get_average_star_zero[:,0], label='Star 0')
    plt.plot(episodes, get_average_dynamic[:,0], label='Dynamic')
    plt.plot(episodes, get_average_star_six[:,0], label='Star 6')
    plt.plot(episodes, get_average_line_gamma_10[:,0], label='Line Gamma 10')
    plt.plot(episodes, get_average_fully_connected[:,0], label='Fully Connected')
    plt.plot(episodes, get_average_star_zero_two[:,0], label='Star Gamma 2')

    plt.title(f'Evaluation of best on switching Agents 0 & 6.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs_twelve/evaluation_switch_best_0_episode_{episode_output}.png')

    # i = 1  (Agents 1 and 4)
    plt.figure()
    plt.plot(episodes, get_average_random_switch[:, 1], label = f'Random')
    plt.plot(episodes, get_average_star_four[:,1], label='Star 4')
    plt.plot(episodes, get_average_dynamic[:,1], label='Dynamic')
    plt.plot(episodes, get_average_star_one[:,1], label='Star 1')
    plt.plot(episodes, get_average_ring_six[:,1], label='Ring 6')
    plt.plot(episodes, get_average_ring_eight[:,1], label='Ring 8')
    plt.plot(episodes, get_average_fully_connected[:,1], label='Fully Connected')
    plt.plot(episodes, get_average_ring_ten[:,1], label='Ring 10')
    plt.plot(episodes, get_average_line_gamma_10[:,1], label='Line Gamma 10')
    plt.plot(episodes, get_average_star_zero_two[:,1], label='Star Gamma 2')

    plt.title(f'Evaluation of best on switching Agents 1 & 4.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs_twelve/evaluation_switch_best_1_episode_{episode_output}.png')

    # i = 2 (Agents 2 and 10)
    plt.figure()
    plt.plot(episodes, get_average_random_switch[:, 2], label = f'Random')
    plt.plot(episodes, get_average_star_two[:,2], label='Star 2')
    plt.plot(episodes, get_average_dynamic[:,2], label='Dynamic') 
    plt.plot(episodes, get_average_star_ten[:,2], label='Star 10')
    plt.plot(episodes, get_average_line_gamma_10[:,2], label='Line Gamma 10')
    plt.plot(episodes, get_average_fully_connected[:,2], label='Fully Connected')
    plt.plot(episodes, get_average_ring_eight[:,2], label='Ring 8')
    plt.plot(episodes, get_average_star_zero_two[:,2], label='Star Gamma 2')
    plt.plot(episodes, get_average_ring_ten[:,2], label='Ring 10')

    plt.title(f'Evaluation of best on switching Agents 2 & 10.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs_twelve/evaluation_switch_best_2_episode_{episode_output}.png')

    # i = 3 (Agents 7 and 8)
    plt.figure()
    plt.plot(episodes, get_average_random_switch[:, 3], label = f'Random')
    plt.plot(episodes, get_average_star_seven[:,3], label='Star 7')
    plt.plot(episodes, get_average_dynamic[:,3], label='Dynamic') 
    plt.plot(episodes, get_average_star_eight[:,3], label='Star 8')
    plt.plot(episodes, get_average_ring_two[:,3], label='Ring 2')
    plt.plot(episodes, get_average_ring_eight[:,3], label='Ring 8')
    plt.plot(episodes, get_average_ring_ten[:,3], label='Ring 10')
    plt.plot(episodes, get_average_ring_six[:,3], label='Ring 6')
    plt.plot(episodes, get_average_ring_four[:,3], label='Ring 4')
    plt.plot(episodes, get_average_fully_connected[:,3], label='Fully Connected')
    plt.plot(episodes, get_average_star_zero_two[:,3], label='Star Gamma 2')
    plt.plot(episodes, get_average_line_gamma_10[:,3], label='Line Gamma 10')

    plt.title(f'Evaluation of best on switching Agents 7 & 8.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs_twelve/evaluation_switch_best_3_episode_{episode_output}.png')

    # i = 4 (Agents 3 and 5)
    plt.figure()
    plt.plot(episodes, get_average_random_switch[:, 4], label = f'Random')
    plt.plot(episodes, get_average_star_three[:,4], label='Star 3')
    plt.plot(episodes, get_average_dynamic[:,4], label='Dynamic') 
    plt.plot(episodes, get_average_star_five[:,4], label='Star 5')
    plt.plot(episodes, get_average_ring_eight[:,4], label='Ring 8')
    plt.plot(episodes, get_average_star_zero_two[:,4], label='Star Gamma 2')
    plt.plot(episodes, get_average_ring_six[:,4], label='Ring 6')
    plt.plot(episodes, get_average_line_gamma_10[:,4], label='Line Gamma 10')
    plt.plot(episodes, get_average_fully_connected[:,4], label='Fully Connected')
    plt.plot(episodes, get_average_ring_ten[:,4], label='Ring 10')
    plt.plot(episodes, get_average_ring_four[:,4], label='Ring 4')

    plt.title(f'Evaluation of best on switching Agents 3 & 5.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs_twelve/evaluation_switch_best_4_episode_{episode_output}.png')

    # i = 5 (Agents 9 and 11)
    plt.figure()
    plt.plot(episodes, get_average_random_switch[:, 5], label = f'Random')
    plt.plot(episodes, get_average_dynamic[:,5], label='Dynamic') 
    plt.plot(episodes, get_average_star_nine[:,5], label='Star 9')
    plt.plot(episodes, get_average_star_eleven[:,5], label='Star 11')
    plt.plot(episodes, get_average_line_gamma_10[:,5], label='Line Gamma 10')
    plt.plot(episodes, get_average_ring_six[:,5], label='Ring 6')
    plt.plot(episodes, get_average_fully_connected[:,5], label='Fully Connected')
    plt.plot(episodes, get_average_star_zero_two[:,5], label='Star Gamma 2')
    plt.plot(episodes, get_average_ring_eight[:,5], label='Ring 8') 
    plt.plot(episodes, get_average_ring_ten[:,5], label='Ring 10')
    plt.plot(episodes, get_average_ring_four[:,5], label='Ring 4')

    plt.title(f'Evaluation of best on switching Agents 9 & 11.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs_twelve/evaluation_switch_best_5_episode_{episode_output}.png')


def create_ring(episode_output):

    """
    Creates the graphs for Ring graphs
    episode_output - How often the data should be graphed
    """

    FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_10000_0_'
    # RANDOM_PART = 'saved_data_twelve/AgentType.RANDOM_12_10_10000_0_'
    RANDOM_PART = 'saved_data_twelve/AgentType.EB_Lidard_12_10_1000_0_'
    

    filename = RANDOM_PART+'0.pkl'
    episodes = [file for file in _pickle_loader(filename)][0]['reward'][2][::episode_output] 

    random_switch = [0]
    ring_two = [4]
    ring_four = [5]
    ring_six = [6]
    ring_eight = [7]
    ring_ten = [8]

    get_average_random_switch = get_average_values(random_switch, RANDOM_PART)[::episode_output]
    get_average_ring_two = get_average_values(ring_two, FIRST_PART)[::episode_output]
    get_average_ring_four = get_average_values(ring_four, FIRST_PART)[::episode_output]
    get_average_ring_six = get_average_values(ring_six, FIRST_PART)[::episode_output]
    get_average_ring_eight = get_average_values(ring_eight, FIRST_PART)[::episode_output]
    get_average_ring_ten = get_average_values(ring_ten, FIRST_PART)[::episode_output]

    for i in range(6):
        plt.figure()
        plt.plot(episodes, get_average_random_switch[:, i], label = f'Random')
        plt.plot(episodes, get_average_ring_two[:,i], label='Ring 2')
        plt.plot(episodes, get_average_ring_four[:,i], label='Ring 4')
        plt.plot(episodes, get_average_ring_six[:,i], label='Ring 6')
        plt.plot(episodes, get_average_ring_eight[:,i], label='Ring 8')
        plt.plot(episodes, get_average_ring_ten[:,i], label='Ring 10')

    
        if i == 0:
            plt.title(f'Evaluation of Ring on switching Agents 0 & 6.  ')

        if i == 1:
            plt.title(f'Evaluation of Ring on switching Agents 1 & 4.  ')

        if i == 2:
            plt.title(f'Evaluation of Ring on switching Agents 2 & 10.  ')

        if i == 3:
            plt.title(f'Evaluation of Ring on switching Agents 7 & 8.  ')

        if i == 4:
            plt.title(f'Evaluation of Ring on switching Agents 3 & 5.  ')
        
        if i == 5:
            plt.title(f'Evaluation of Ring on switching Agents 9 & 11. ')

        plt.ylabel("Reward")
        plt.xlabel("Episode number")
        plt.legend()
        plt.savefig(f'figs_twelve/evaluation_switch_Ring_{i}_episode_{episode_output}.png')


def create_star(episode_output):

    """
    Creates the graphs for Star graphs
    episode_output - How often the data should be graphed
    """

    FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_10000_0_'
    # RANDOM_PART = 'saved_data_twelve/AgentType.RANDOM_12_10_10000_0_'
    RANDOM_PART = 'saved_data_twelve/AgentType.EB_Lidard_12_10_1000_0_'
    

    filename = RANDOM_PART+'0.pkl'
    episodes = [file for file in _pickle_loader(filename)][0]['reward'][2][::episode_output]   

    random_switch = [0]
    star_zero = [9]
    star_zero_two = [10]
    star_one = [11]
    star_two = [12]
    star_three = [13]
    star_four = [14]
    star_five = [15]
    star_six = [16]
    star_seven = [17]
    star_eight = [18]
    star_nine = [19]
    star_ten = [20]
    star_eleven = [21]

    get_average_random_switch = get_average_values(random_switch, RANDOM_PART)[::episode_output]
    get_average_star_zero = get_average_values(star_zero, FIRST_PART)[::episode_output]
    get_average_star_one = get_average_values(star_one, FIRST_PART)[::episode_output]
    get_average_star_two = get_average_values(star_two, FIRST_PART)[::episode_output]
    get_average_star_three = get_average_values(star_three, FIRST_PART)[::episode_output]
    get_average_star_four = get_average_values(star_four, FIRST_PART)[::episode_output]
    get_average_star_five = get_average_values(star_five, FIRST_PART)[::episode_output]
    get_average_star_six = get_average_values(star_six, FIRST_PART)[::episode_output]
    get_average_star_seven = get_average_values(star_seven, FIRST_PART)[::episode_output]
    get_average_star_eight = get_average_values(star_eight, FIRST_PART)[::episode_output]
    get_average_star_nine = get_average_values(star_nine, FIRST_PART)[::episode_output]
    get_average_star_ten = get_average_values(star_ten, FIRST_PART)[::episode_output]
    get_average_star_eleven = get_average_values(star_eleven, FIRST_PART)[::episode_output]
    get_average_star_zero_two = get_average_values(star_zero_two, FIRST_PART)[::episode_output]

    for i in range(6):
        plt.figure()
        plt.plot(episodes, get_average_random_switch[:, i], label = f'Random')
        plt.plot(episodes, get_average_star_zero[:,i], label='Star 0')
        plt.plot(episodes, get_average_star_zero_two[:,i], label='Star 0 (Gamma 2)')
        plt.plot(episodes, get_average_star_one[:,i], label='Star 1')
        plt.plot(episodes, get_average_star_two[:,i], label='Star 2')
        plt.plot(episodes, get_average_star_three[:,i], label='Star 3')
        plt.plot(episodes, get_average_star_four[:,i], label='Star 4')
        plt.plot(episodes, get_average_star_five[:,i], label='Star 5')
        plt.plot(episodes, get_average_star_six[:,i], label='Star 6')
        plt.plot(episodes, get_average_star_seven[:,i], label='Star 7')
        plt.plot(episodes, get_average_star_eight[:,i], label='Star 8')
        plt.plot(episodes, get_average_star_nine[:,i], label='Star 9')
        plt.plot(episodes, get_average_star_ten[:,i], label='Star 10')
        plt.plot(episodes, get_average_star_eleven[:,i], label='Star 11')

    
        if i == 0:
            plt.title(f'Evaluation of Star on switching Agents 0 & 6. ')

        if i == 1:
            plt.title(f'Evaluation of Star on switching Agents 1 & 4. ')

        if i == 2:
            plt.title(f'Evaluation of Star on switching Agents 2 & 10. ')

        if i == 3:
            plt.title(f'Evaluation of Star on switching Agents 7 & 8. ')

        if i == 4:
            plt.title(f'Evaluation of Star on switching Agents 3 & 5. ')
        
        if i == 5:
            plt.title(f'Evaluation of Star on switching Agents 9 & 11. ')

        plt.ylabel("Reward")
        plt.xlabel("Episode number")
        plt.legend()
        plt.savefig(f'figs_twelve/evaluation_switch_Star_{i}_episode_{episode_output}.png')



def main_twelve(episode_output):

    """
    Creates the graphs for All graphs dealing with 12 agents
    episode_output - How often the data should be graphed
    """

    FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_10000_0_'
    # RANDOM_PART = 'saved_data_twelve/AgentType.RANDOM_12_10_10000_0_'
    RANDOM_PART = 'saved_data_twelve/AgentType.EB_Lidard_12_10_1000_0_'
    

    filename = FIRST_PART+'0.pkl'
    episodes = [file for file in _pickle_loader(filename)][0]['reward'][2][::episode_output]

    random_switch = [0]
    dynamic = [0]
    line_gamma_0 = [1]
    line_gamma_10 = [2]
    fully_connected = [3]
    ring_two = [4]
    ring_four = [5]
    ring_six = [6]
    ring_eight = [7]
    ring_ten = [8]
    star_zero = [9]
    star_zero_two = [10]
    star_one = [11]
    star_two = [12]
    star_three = [13]
    star_four = [14]
    star_five = [15]
    star_six = [16]
    star_seven = [17]
    star_eight = [18]
    star_nine = [19]
    star_ten = [20]
    star_eleven = [21]

    get_average_random_switch = get_average_values(random_switch, RANDOM_PART)[::episode_output]
    get_average_dynamic = get_average_values(dynamic, FIRST_PART)[::episode_output]
    get_average_line_gamma_0 = get_average_values(line_gamma_0, FIRST_PART)[::episode_output]
    get_average_line_gamma_10 = get_average_values(line_gamma_10, FIRST_PART)[::episode_output]
    get_average_fully_connected = get_average_values(fully_connected, FIRST_PART)[::episode_output]
    get_average_ring_two = get_average_values(ring_two, FIRST_PART)[::episode_output]
    get_average_ring_four = get_average_values(ring_four, FIRST_PART)[::episode_output]
    get_average_ring_six = get_average_values(ring_six, FIRST_PART)[::episode_output]
    get_average_ring_eight = get_average_values(ring_eight, FIRST_PART)[::episode_output]
    get_average_ring_ten = get_average_values(ring_ten, FIRST_PART)[::episode_output]
    get_average_star_zero = get_average_values(star_zero, FIRST_PART)[::episode_output]
    get_average_star_one = get_average_values(star_one, FIRST_PART)[::episode_output]
    get_average_star_two = get_average_values(star_two, FIRST_PART)[::episode_output]
    get_average_star_three = get_average_values(star_three, FIRST_PART)[::episode_output]
    get_average_star_four = get_average_values(star_four, FIRST_PART)[::episode_output]
    get_average_star_five = get_average_values(star_five, FIRST_PART)[::episode_output]
    get_average_star_six = get_average_values(star_six, FIRST_PART)[::episode_output]
    get_average_star_seven = get_average_values(star_seven, FIRST_PART)[::episode_output]
    get_average_star_eight = get_average_values(star_eight, FIRST_PART)[::episode_output]
    get_average_star_nine = get_average_values(star_nine, FIRST_PART)[::episode_output]
    get_average_star_ten = get_average_values(star_ten, FIRST_PART)[::episode_output]
    get_average_star_eleven = get_average_values(star_eleven, FIRST_PART)[::episode_output]
    get_average_star_zero_two = get_average_values(star_zero_two, FIRST_PART)[::episode_output]


    for i in range(6):
        plt.figure()
        plt.plot(episodes, get_average_random_switch[:, i], label = f'Random')
        plt.plot(episodes, get_average_dynamic[:, i], label='Dynamic')
        plt.plot(episodes, get_average_line_gamma_0[:, i], label='0')
        plt.plot(episodes, get_average_line_gamma_10[:, i], label='10')
        plt.plot(episodes, get_average_fully_connected[:, i], label='FC')
        plt.plot(episodes, get_average_ring_two[:, i], label='Ring 2')
        plt.plot(episodes, get_average_ring_four[:, i], label='Ring 4')
        plt.plot(episodes, get_average_ring_six[:, i], label='Ring 6')
        plt.plot(episodes, get_average_ring_eight[:, i], label='Ring 8')
        plt.plot(episodes, get_average_ring_ten[:, i], label='Ring 10')
        plt.plot(episodes, get_average_star_zero[:, i], label='Star 0')
        plt.plot(episodes, get_average_star_zero_two[:, i], label='Star 0 (Gamma 2)')
        plt.plot(episodes, get_average_star_one[:, i], label='Star 1')
        plt.plot(episodes, get_average_star_two[:, i], label='Star 2')
        plt.plot(episodes, get_average_star_three[:,i], label='Star 3')
        plt.plot(episodes, get_average_star_four[:,i], label='Star 4')
        plt.plot(episodes, get_average_star_five[:,i], label='Star 5')
        plt.plot(episodes, get_average_star_six[:,i], label='Star 6')
        plt.plot(episodes, get_average_star_seven[:,i], label='Star 7')
        plt.plot(episodes, get_average_star_eight[:,i], label='Star 8')
        plt.plot(episodes, get_average_star_nine[:,i], label='Star 9')
        plt.plot(episodes, get_average_star_ten[:,i], label='Star 10')
        plt.plot(episodes, get_average_star_eleven[:,i], label='Star 11')

    
        if i == 0:
            plt.title(f'Evaluation on switching Agents 0 & 6. ')

        if i == 1:
            plt.title(f'Evaluation on switching Agents 1 & 4. ')

        if i == 2:
            plt.title(f'Evaluation on switching Agents 2 & 10. ')

        if i == 3:
            plt.title(f'Evaluation on switching Agents 7 & 8. ')

        if i == 4:
            plt.title(f'Evaluation on switching Agents 3 & 5. ')
        
        if i == 5:
            plt.title(f'Evaluation on switching Agents 9 & 11. ')

        plt.ylabel("Reward")
        plt.xlabel("Episode number")
        plt.legend()
        plt.savefig(f'figs_twelve/evaluation_switch_{i}_episode_{episode_output}.png')

    plt.close()


def create_basic(episode_output):

    """
    Creates the graphs for 'Basic' graphs
    episode_output - How often the data should be graphed
    """

    FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_10000_0_'
    # RANDOM_PART = 'saved_data_twelve/AgentType.RANDOM_12_10_10000_0_'
    RANDOM_PART = 'saved_data_twelve/AgentType.EB_Lidard_12_10_1000_0_'
    

    filename = RANDOM_PART+'0.pkl'
    episodes = [file for file in _pickle_loader(filename)][0]['reward'][2][::episode_output]

   
    random_switch = [0]
    dynamic = [0]
    line_gamma_0 = [1]
    line_gamma_10 = [2]
    fully_connected = [3]

    get_average_random_switch = get_average_values(random_switch, RANDOM_PART)[::episode_output]
    get_average_dynamic = get_average_values(dynamic, FIRST_PART)[::episode_output]
    get_average_line_gamma_0 = get_average_values(line_gamma_0, FIRST_PART)[::episode_output]
    get_average_line_gamma_10 = get_average_values(line_gamma_10, FIRST_PART)[::episode_output]
    get_average_fully_connected = get_average_values(fully_connected, FIRST_PART)[::episode_output]


    for i in range(6):
        plt.figure()
        plt.plot(episodes, get_average_random_switch[:, i], label = f'Random')
        plt.plot(episodes, get_average_dynamic[:,i], label='Dynamic')
        plt.plot(episodes, get_average_line_gamma_0[:,i], label='0')
        plt.plot(episodes, get_average_line_gamma_10[:,i], label='10')
        plt.plot(episodes, get_average_fully_connected[:,i], label='FC')


    
        if i == 0:
            plt.title(f'Evaluation on switching Agents 0 & 6. ')

        if i == 1:
            plt.title(f'Evaluation on switching Agents 1 & 4. ')

        if i == 2:
            plt.title(f'Evaluation on switching Agents 2 & 10. ')

        if i == 3:
            plt.title(f'Evaluation on switching Agents 7 & 8. ')

        if i == 4:
            plt.title(f'Evaluation on switching Agents 3 & 5. ')
        
        if i == 5:
            plt.title(f'Evaluation on switching Agents 9 & 11. ')

        plt.ylabel("Reward")
        plt.xlabel("Episode number")
        plt.legend()
        plt.savefig(f'figs_twelve/evaluation_switch_basic_{i}_episode_{episode_output}.png')

    plt.close()


def main_four(episode_output):

    """
    Creates the graphs for all graphs for 4 agents
    episode_output - How often the data should be graphed
    """

    FIRST_PART = 'saved_data/AgentType.ORIGINAL_4_10_10000_0_'
    FIRST_PART_IQL = 'saved_data/AgentType.IQL_4_10_10000_0_'
    # RANDOM_PART = 'saved_data/AgentType.RANDOM_4_10_10000_0_'
    RANDOM_PART = 'saved_data_twelve/AgentType.EB_Lidard_12_10_1000_0_'
    

    # This is the different tests which have been done and the data saved
    same_original_no = [0]  
    same_original_full = [1] 
    switch_original_full = [3]
    switch_original_no = [2] 
    same_random = [0]
    switch_random = [1]
    same_iql = [0]
    switch_iql = [1]

    filename = FIRST_PART+'0.pkl'       # Pick num from gamma hop
    episodes = [file for file in _pickle_loader(filename)][0]['reward'][2][::episode_output]    

    random_average = get_average_values(same_random, RANDOM_PART)
    iql_average = get_average_values(same_iql, FIRST_PART_IQL)
    original_full_average = get_average_values(same_original_full, FIRST_PART)
    original_none_average = get_average_values(same_original_no, FIRST_PART)

    # print(original_full_average)


    plt.figure()
    plt.plot(episodes, random_average[::episode_output], label = 'Random')
    plt.plot(episodes, iql_average[::episode_output], label = 'IQL')
    plt.plot(episodes, original_full_average[::episode_output], label = 'Full Comm')
    plt.plot(episodes, original_none_average[::episode_output], label = 'No Comm')
    
    
    
    plt.title(f'Evaluation - Same position.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs/evaluation_combined_same_{episode_output}.png')

    plt.figure()

    random_switch_average = get_average_values(switch_random, RANDOM_PART)
    switch_iql_average = get_average_values(switch_iql, FIRST_PART_IQL)
    original_full_switch_average = get_average_values(switch_original_full, FIRST_PART)
    original_none_switch_average = get_average_values(switch_original_no, FIRST_PART)

    plt.plot(episodes, random_switch_average[::episode_output], label = 'Random')
    plt.plot(episodes, switch_iql_average[::episode_output], label = 'IQL')
    plt.plot(episodes, original_full_switch_average[::episode_output], label = 'Full Comm')
    plt.plot(episodes, original_none_switch_average[::episode_output], label = 'No Comm')
    
    
    plt.title(f'Evaluation - Switch position.  ')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.legend()
    plt.savefig(f'figs/evaluation_combined_switch_{episode_output}.png')


def average_over_10(nums, first_part):

    """
    This averages out the rewards over 10 episodes
    """
    
    open_files = _open(nums, first_part)

    open_files = create_array(open_files)
    first_part = []

    total = 0
    for i in range(9):
        total += open_files[i]
        average = total/(i+1)
        first_part.append(average)

   # print(len(first_part))

    final_array = first_part+[sum(open_files[i-10:i])/10 for i in range(10, len(open_files)+1, 1)]
    
    for i in range(len(final_array)):
        if final_array[i] > -0.5:
   #         print(final_array[i], i)
            pass
    
    return final_array


main()
