import numpy as np
import math
from utils import get_average_values

def main():
    print("Choose an option:\n "
          "1. Create rankings\n"
          )
    
    option = int(input())
    if option == 1:
        results()
    else:
        print("Not an option")

def create_results(num_of_episodes):

    """
    This will actually calculate the rankings
    num_of_episodes - How many episodes to take into consideration - EG 100 will take the last 100 episodes
    """

    # FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_10000_0_'
    FIRST_PART = 'saved_data_twelve/AgentType.ORIGINAL_12_10_100_0_' #100 Episodes for testing


    # These are where the data is stored on my machine
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

    # Gets the average values needed
    get_average_dynamic = get_average_values(dynamic, FIRST_PART)
    get_average_line_gamma_0 = get_average_values(line_gamma_0, FIRST_PART)
    get_average_line_gamma_10 = get_average_values(line_gamma_10, FIRST_PART)
    get_average_fully_connected = get_average_values(fully_connected, FIRST_PART)
    get_average_ring_two = get_average_values(ring_two, FIRST_PART)
    get_average_ring_four = get_average_values(ring_four, FIRST_PART)
    get_average_ring_six = get_average_values(ring_six, FIRST_PART)
    get_average_ring_eight = get_average_values(ring_eight, FIRST_PART)
    get_average_ring_ten = get_average_values(ring_ten, FIRST_PART)
    get_average_star_zero = get_average_values(star_zero, FIRST_PART)
    get_average_star_one = get_average_values(star_one, FIRST_PART)
    get_average_star_two = get_average_values(star_two, FIRST_PART)
    get_average_star_three = get_average_values(star_three, FIRST_PART)
    get_average_star_four = get_average_values(star_four, FIRST_PART)
    get_average_star_five = get_average_values(star_five, FIRST_PART)
    get_average_star_six = get_average_values(star_six, FIRST_PART)
    get_average_star_seven = get_average_values(star_seven, FIRST_PART)
    get_average_star_eight = get_average_values(star_eight, FIRST_PART)
    get_average_star_nine = get_average_values(star_nine, FIRST_PART)
    get_average_star_ten = get_average_values(star_ten, FIRST_PART)
    get_average_star_eleven = get_average_values(star_eleven, FIRST_PART)
    get_average_star_zero_two = get_average_values(star_zero_two, FIRST_PART)
    # get_average_random = get_average_values([0], 'saved_data_twelve/AgentType.RANDOM_12_10_10000_0_')
    get_average_random = get_average_values([0], 'saved_data_twelve/AgentType.RANDOM_12_10_100_0_') #100 episodes for testing

    # Creates the Mean and SD
    dynamic_mean, dynamic_sd = create_average(get_average_dynamic, num_of_episodes, 'Dynamic')
    line_gamma_0_mean, line_gamma_0_sd = create_average(get_average_line_gamma_0, num_of_episodes, 'Line Gamma 0')
    line_gamma_10_mean, line_gamma_10_sd = create_average(get_average_line_gamma_10, num_of_episodes, 'Line Gamma 10')
    fc_mean, fc_sd = create_average(get_average_fully_connected, num_of_episodes, 'Fully Connected')
    ring_two_mean, ring_two_sd = create_average(get_average_ring_two, num_of_episodes, 'Ring Two')
    ring_four_mean, ring_four_sd = create_average(get_average_ring_four, num_of_episodes, 'Ring Four')
    ring_six_mean, ring_six_sd = create_average(get_average_ring_six, num_of_episodes, 'Ring Six')
    ring_eight_mean, ring_eight_sd = create_average(get_average_ring_eight, num_of_episodes, 'Ring Eight')
    ring_ten_mean, ring_ten_sd = create_average(get_average_ring_ten, num_of_episodes, 'Ring Ten')
    star_zero_mean, star_zero_sd = create_average(get_average_star_zero, num_of_episodes, 'Star Centre Zero')
    star_one_mean, star_one_sd = create_average(get_average_star_one, num_of_episodes, 'Star Centre One')
    star_two_mean, star_two_sd = create_average(get_average_star_two, num_of_episodes, 'Star Centre Two')
    star_three_mean, star_three_sd = create_average(get_average_star_three, num_of_episodes, 'Star Centre Three')
    star_four_mean, star_four_sd = create_average(get_average_star_four, num_of_episodes, 'Star Centre Four')
    star_five_mean, star_five_sd = create_average(get_average_star_five, num_of_episodes, 'Star Centre Five')
    star_six_mean, star_six_sd = create_average(get_average_star_six, num_of_episodes, 'Star Centre Six')
    star_seven_mean, star_seven_sd = create_average(get_average_star_seven, num_of_episodes, 'Star Centre Seven')
    star_eight_mean, star_eight_sd = create_average(get_average_star_eight, num_of_episodes, 'Star Centre Eight')
    star_nine_mean, star_nine_sd = create_average(get_average_star_nine, num_of_episodes, 'Star Centre Nine')
    star_ten_mean, star_ten_sd = create_average(get_average_star_ten, num_of_episodes, 'Star Centre Ten')
    star_eleven_mean, star_eleven_sd = create_average(get_average_star_eleven, num_of_episodes, 'Star Centre Eleven')
    star_zero_two_mean, star_zero_two_sd = create_average(get_average_star_zero_two, num_of_episodes, 'Star Centre Zero Gamma Two')
    random_mean, random_sd = create_average(get_average_random, num_of_episodes, 'Random')

    mean_data_dict = {'Dynamic': dynamic_mean, 'Line Gamma 0': line_gamma_0_mean, 'Line Gamma 10': line_gamma_10_mean, 'Fully Connected': fc_mean, 
                       'Ring Two': ring_two_mean, 'Ring Four': ring_four_mean, 'Ring Six': ring_six_mean, 'Ring Eight': ring_eight_mean, 'Ring Ten':ring_ten_mean,
                       'Star Centre Zero': star_zero_mean, 'Star Centre One': star_one_mean, 'Star Centre Two': star_two_mean, 'Star Centre Three': star_three_mean,
                       'Star Centre Four': star_four_mean, 'Star Centre Five': star_five_mean, 'Star Centre Six': star_six_mean, 'Star Centre Seven': star_seven_mean,
                       'Star Centre Eight': star_eight_mean, 'Star Centre Nine': star_nine_mean, 'Star Centre Ten': star_ten_mean, 'Star Centre Eleven': star_eleven_mean,
                       'Star Centre Zero Gamma Two': star_zero_two_mean, 'Random': random_mean}
    

    return rankings(mean_data_dict)


def rankings(mean_data_dict):

    """This will create the rankings
    mean_data_dict - Contains the mean data of each of the different graphs tested"""

    swap_zero_six = []
    swap_one_four = []
    swap_two_ten = []
    swap_seven_eight = []
    swap_three_five = []
    swap_nine_eleven = []

    # Adds the mean data for each of the swaps for each graph into the correct list
    for mean_data in mean_data_dict.values():
        for i, mean in enumerate(mean_data[0]):
            if i == 0:
                swap_zero_six.append((mean, mean_data[1]))

            elif i == 1:
                swap_one_four.append((mean, mean_data[1]))

            elif i == 2:
                swap_two_ten.append((mean, mean_data[1]))

            elif i == 3:
                swap_seven_eight.append((mean, mean_data[1]))

            elif i == 4:
                swap_three_five.append((mean, mean_data[1]))
            
            elif i == 5:
                swap_nine_eleven.append((mean, mean_data[1]))
    
    swap_zero_six.sort(reverse=True)
    swap_one_four.sort(reverse=True)
    swap_two_ten.sort(reverse=True)
    swap_seven_eight.sort(reverse=True)
    swap_three_five.sort(reverse=True)
    swap_nine_eleven.sort(reverse=True)

    swap_arrays = [(swap_zero_six, 'Switching Agents 0 & 6'), (swap_one_four, 'Switching Agents 1 & 4'), (swap_two_ten, 'Switching Agents 2 & 10'),
                   (swap_seven_eight, 'Switching Agents 7 & 8'), (swap_three_five, 'Switching Agents 3 & 5'), (swap_nine_eleven, 'Switching Agents 9 & 11')]
    positions = {'Dynamic': [0, 0], 'Line Gamma 0': [0, 0], 'Line Gamma 10': [0, 0], 'Fully Connected': [0, 0], 'Ring Two': [0, 0], 'Ring Four': [0, 0],
                 'Ring Six': [0, 0], 'Ring Eight': [0, 0], 'Ring Ten': [0, 0], 'Star Centre Zero': [0, 0], 'Star Centre One': [0, 0], 
                 'Star Centre Two': [0, 0], 'Star Centre Three': [0, 0], 'Star Centre Four': [0, 0], 'Star Centre Five': [0, 0], 'Star Centre Six': [0, 0], 
                 'Star Centre Seven': [0, 0], 'Star Centre Eight': [0, 0], 'Star Centre Nine': [0, 0], 'Star Centre Ten': [0, 0], 'Star Centre Eleven': [0, 0],
                 'Star Centre Zero Gamma Two': [0, 0], 'Random': [0, 0]}
    
    # Prints the lists and creates the position rankings
    for swap_data in swap_arrays:
        print_pretty(swap_data)
        print('\n')
        for position, graph in enumerate(swap_data[0]):
            positions[graph[1]][0] += (position+1)
            positions[graph[1]][1] += (graph[0])
    
    rankings = []
    for key, value in positions.items():
        rankings.append(((value[0]/6, value[1]/6), key))
    
    rankings.sort(key=lambda x: x[0][0])
    final_rankings_pos = (rankings, 'Final Rankings on average position')
    print_pretty(final_rankings_pos)
    print('\n')
    
    rankings.sort(key=lambda x: x[0][1], reverse=True)
    final_rankings_score = (rankings, 'Final Rankings on average reward')
    print_pretty(final_rankings_score)

    return final_rankings_pos, final_rankings_score


def print_pretty(tuple_of_data):
    '''
    Prints the data out in a pretty manner
    tuple_of_data - Data to be shown should be sorted and tuple_of_data like: ([(data, data_name)], 'Name') 
    '''

    print(f'Printing the rankings of {tuple_of_data[1]}')
    print('Position: Graph, Ranked on')
    for i, data in enumerate(tuple_of_data[0]):
        print(f'{i+1}. {data[1]}, {data[0]}')


def create_average(average_values, num_of_episodes, test):

    """
    Creates the average and SD of the data
    average_values - the episode data to work on
    num_of_episodes - The number of episodes
    test - Which graph is being checked
    """

    summed_values = np.zeros(6)
    for i in range(num_of_episodes):
        summed_values += average_values[(i*-1)+-1]

    mean = summed_values/num_of_episodes
    sd_part_one = np.zeros(6)
    for i in range(num_of_episodes):
        sd_part_one += (average_values[(i*-1)+-1]-mean)**2
    sd_part_two = (1/num_of_episodes) * sd_part_one
    sd = np.zeros(6)
    for i in range(6):
        sd[i] = math.sqrt(sd_part_two[i])
    
    return [mean, test], [sd, test]

def results():

    """
    Starts the program
    """

    print("Choose how many episodes to use for the mean and SD analysis:")
    choice = int(input())
    return create_results(choice)


main()