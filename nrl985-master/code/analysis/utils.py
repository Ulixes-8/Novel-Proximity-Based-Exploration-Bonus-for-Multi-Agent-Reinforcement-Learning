import dill


def _pickle_loader(filename):
    
    with open(filename, 'rb') as f:
        while True:
            try:
                yield dill.load(f)
            except EOFError:
                break


def _open(nums, first_part):

    """
    Opens up the data
    """

    open_files = []
    for value in nums:
        filename = f'{first_part}{value}.pkl'
        # This gets the evaluation reward
        array = [file for file in _pickle_loader(filename)][0]['reward'][1]
        open_files.append(array)
    
    return open_files



def get_average_values(nums, first_part):

    """
    This gets the evaluation values
    """

    open_files = _open(nums, first_part)

    return create_array(open_files)



def create_array(arrays_to_use):

    """
    Creates the array
    """
    #print(arrays_to_use[0][50])
    #print(arrays_to_use)
    final_array = sum(arrays_to_use)
   

    # print(final_array)
    #print(final_array/len(arrays_to_use))

    return final_array/len(arrays_to_use)

