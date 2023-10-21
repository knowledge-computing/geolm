import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_self_closest_match(sim_matrix, word_list):
    '''sim_matrix should be (n,n)'''
    n = sim_matrix.shape[0]
    sim_matrix[range(n), range(n)] = 0
    indices = np.argmax(sim_matrix, axis = -1)
    ret_list = []
    for ind in indices:
        ret_list.append(word_list[ind])
    return ret_list


def find_ref_closest_match(sim_matrix, word_list):
    '''
    sim_matrix should be (n_ref, n_query)
    word_list should be (n_ref,)
    '''
    n_ref, n_query = sim_matrix.shape[0], sim_matrix.shape[1]
    indices = np.argmax(sim_matrix, axis = 0) # similarity matrix, take the maximum
    #print(indices)
    ret_list = []
    for ind in indices:
        ret_list.append(word_list[ind])
    return ret_list

def sort_ref_closest_match(sim_matrix, word_list):
    '''
    sim_matrix should be (n_ref, n_query)
    word_list should be (n_ref,)
    '''
    n_ref, n_query = sim_matrix.shape[0], sim_matrix.shape[1]
    
    indices_list = np.argsort(sim_matrix, axis = 0)[::-1] # descending order
    
    #print(indices_list)
    ret_list = []
    for indices in indices_list:
        word_sorted = []
        for ind in indices:
            word_sorted.append(word_list[ind])
        ret_list.append(word_sorted)
    return ret_list