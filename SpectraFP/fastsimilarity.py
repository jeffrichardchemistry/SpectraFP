import numpy as np
import pandas as pd
#from numba import prange, jit,njit,generated_jit
from numpy import sqrt

#@njit(fastmath=True, cache=True,nogil=True)

def calc_similarity(similarity_metric='tanimoto', alpha=1, beta=1,
                    base_train=np.array([[1,0,1,0,0], [1,0,1,1,1], [1,0,1,0,1]], dtype='uint32'),
                    base_test=np.array([[1,0,1,0,1]], dtype='uint32')):
    
    l_base_train = base_train.shape[0]
    l_base_test = base_test.shape[0]
    
    result = []
    
    if similarity_metric == 'tanimoto':
        for i_test in range(l_base_test):
            get_sims = []
            for i_train in range(l_base_train):
                A = base_test[i_test] 
                B = base_train[i_train]

                AnB = A & B #intersection
                onlyA = B < A #A is a subset of B
                onlyB = A < B #B is a subset of A
                #AuB_0s = A | B #Union (for count de remain zeros)

                sim = AnB.sum() / (onlyA.sum() + onlyB.sum() + AnB.sum())
                get_sims.append(sim)
            result.append(get_sims)        
        return result
    
    if similarity_metric == 'tversky':
        for i_test in range(l_base_test):
            get_sims = []
            for i_train in range(l_base_train):
                A = base_test[i_test] 
                B = base_train[i_train]

                AnB = A & B #intersection
                onlyA = B < A #A is a subset of B
                onlyB = A < B #B is a subset of A
                #AuB_0s = A | B #Union (for count de remain zeros)
                #AuB_0s = np.count_nonzero(AuB_0s==0)

                sim = AnB.sum() / (alpha*onlyA.sum() + beta*onlyB.sum() + AnB.sum())
                get_sims.append(sim)
            result.append(get_sims)        
        return result
    
    if similarity_metric == 'geometric':
        for i_test in range(l_base_test):
            get_sims = []
            for i_train in range(l_base_train):
                A = base_test[i_test] 
                B = base_train[i_train]

                AnB = A & B #intersection
                onlyA = B < A #A is a subset of B
                onlyB = A < B #B is a subset of A
                #AuB_0s = A | B #Union (for count de remain zeros)
                #AuB_0s = np.count_nonzero(AuB_0s==0)

                sim = AnB.sum() / (sqrt((onlyA.sum() + AnB.sum()) * (onlyB.sum() + AnB.sum())))
                get_sims.append(sim)
            result.append(get_sims)        
        return result
    
    if similarity_metric == 'arithmetic':
        for i_test in range(l_base_test):
            get_sims = []
            for i_train in range(l_base_train):
                A = base_test[i_test] 
                B = base_train[i_train]

                AnB = A & B #intersection
                onlyA = B < A #A is a subset of B
                onlyB = A < B #B is a subset of A
                #AuB_0s = A | B #Union (for count de remain zeros)
                #AuB_0s = np.count_nonzero(AuB_0s==0)

                sim = 2*AnB.sum() / ( onlyA.sum() + onlyB.sum() + (2*AnB.sum()) )
                get_sims.append(sim)
            result.append(get_sims)        
        return result
    
    if similarity_metric == 'euclidian':
        for i_test in range(l_base_test):
            get_sims = []
            for i_train in range(l_base_train):
                A = base_test[i_test] 
                B = base_train[i_train]

                AnB = A & B #intersection
                onlyA = B < A #A is a subset of B
                onlyB = A < B #B is a subset of A
                AuB_0s = A | B #Union (for count de remain zeros)
                AuB_0s = np.count_nonzero(AuB_0s==0)

                sim = sqrt((AnB.sum()  + AuB_0s ) / (onlyA.sum()  + onlyB.sum()  + AnB.sum() + AuB_0s))
                get_sims.append(sim)
            result.append(get_sims)        
        return result
    
    if similarity_metric == 'manhattan':
        for i_test in range(l_base_test):
            get_sims = []
            for i_train in range(l_base_train):
                A = base_test[i_test] 
                B = base_train[i_train]

                AnB = A & B #intersection
                onlyA = B < A #A is a subset of B
                onlyB = A < B #B is a subset of A
                AuB_0s = A | B #Union (for count de remain zeros)
                AuB_0s = np.count_nonzero(AuB_0s==0)

                sim = (onlyA.sum() + onlyB.sum()) / (onlyA.sum() + onlyB.sum() + AnB.sum() + AuB_0s)
                get_sims.append(sim)
            result.append(get_sims)        
        return result            


def similarity(base_train, base_test, similarity_metric='tanimoto', alpha=1, beta=1, threshold=0.9):
    """
    This function returns a dictionary with all test samples and yours indexes and
    respectively similarities
    """
    xi_train = base_train.astype('uint64')
    xi_test = base_test.astype('uint64')
    result = calc_similarity(base_train=xi_train, base_test=xi_test,
                         similarity_metric=similarity_metric, alpha=alpha, beta=beta)
    
    result = np.array(result)
    get_all = {}     
    
    for n_matrix,one_matrix in enumerate(result):
        get = {}        
        for i in range(len(one_matrix)):
            if one_matrix[i] >= threshold:
                get[i] = one_matrix[i]
        get_all['Sample_{}'.format(n_matrix)] = get

    return get_all

def getOnematch(base_train, base_test, complete_base,similarity_metric='tanimoto', alpha=1, beta=1, threshold=0.9):
    """
    This function returns the smiles and a dictionary
    with index and similarity value of matches
    Alpha is the parameter of INPUT 
    Beta is the parameter of DATABASE
    """
    get_match = similarity(base_train=base_train, base_test=base_test,
                           similarity_metric=similarity_metric, alpha=alpha, beta=beta, threshold=threshold)
    get_match2 = list(get_match['Sample_0'].keys())    
    smiles = complete_base.iloc[get_match2, 0].values #get only smiles and similarity
    
    n_signs = [spec.sum() for spec in complete_base.iloc[get_match2, 1:].values] #quantity of signs
    #print(get_match, '\n\n', get_match2,'\n')
    #print(n_signs)
    
    union = zip(smiles, get_match['Sample_0'].values(), n_signs)
    output = {}
    outputWn_signs = {}
    for smiles, sim, n_signs in union:
        output[smiles] = sim
        outputWn_signs[smiles] = [sim, n_signs]
    
    #print(outputWn_signs)
    return output,outputWn_signs
    #return smiles, get_match['Sample_0']