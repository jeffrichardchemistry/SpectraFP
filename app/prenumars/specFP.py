import pandas as pd
import numpy as np
from tqdm import tqdm

class SpectraFP():
    def __init__(self, range_spectra = [0,190,0.1]):
        self.start = range_spectra[0]
        self.stop = range_spectra[1]
        self.step = range_spectra[2]
        self._vet_fp = np.arange(self.start, self.stop, self.step)
        self.__x_axis_permitted_ = self._vet_fp.copy()
        self.x_axis_permitted_ = []
        
    def __list2float(self, lista):
        """
        Serve pro caso do meu dataframe onde as listas são uma string.
        Essa função converte pra lista normal
        """
        listfloat = []
        for n in lista.split(','):
            listfloat.append(round(float(n.strip('[').strip(']')), 1))
        
        return listfloat

    def __findPos(self, num ,dec=1):
        """
        Return the ppm positions. We can pass a list of ppm or a single value.
        """  
        lst = []
        array = np.array(0)
        #testando o tipo de dado (lista ou array)
        if type(num) == type(lst) or type(num) == type(array):
            num = [round(x,dec) for x in num] #precisa arredondar pra pegar condição correta
            findedlist = []
            for value_entry in num: #pra cada valor que o usuario entrar vou contar no vetor grande
                for i,vet in enumerate(self._vet_fp):
                    if round(vet, dec) == value_entry:
                        findedlist.append(i)
            return findedlist
                
        else:
            num = round(num, dec)
            findedlist = []
            for i,vet in enumerate(self._vet_fp):
                if round(vet,dec) == num:
                    findedlist.append(i)
            return findedlist
    
    def __permittedPos(self, degree_freedom = 1, axis_range = [], contrast_final = False):     
        """
        A razão pra achar os centroids é impar começando de 3
        por isso usamos (2*degree_freedom+1)
        gf = 1 - 1, 4, 7, 10 --- razao 3
        gf = 2 - 2, 7, 12 --- razao 5
        gf = 3 - 3, 10, 17 -- razao 7
        """
        axis_range = np.arange(axis_range[0], axis_range[1], axis_range[2])#array com os pontos
        all_positions = np.arange(0, len(axis_range), 1) #array com as posições
        
        permit_pos = [] #armazenar as posições dos "centroids"
        centroid = degree_freedom #graus de liberdade
        for pos in all_positions:
            #parar qnd o centroid for maior ou igual a ultima posição do vetor de posições
            if centroid >= all_positions[-1]: 
                break
            permit_pos.append(centroid) #grava a posição do centroid
            #pegar o proximo centroid, a razão é sempre um numero impar
            centroid = centroid + (2*degree_freedom+1)
        
        #quantos grupos de 2*degree+1 consigo formar?
        n_groups = len(axis_range) // (2*degree_freedom+1) #quantos grupos nós temos
        dados_agrupados = n_groups*(2*degree_freedom+1) #quantidade total de valores nesses grupos
        sobra = abs(dados_agrupados - len(axis_range)) #quanto sobra no final que nao estão agrupados
        
        permit_pos = permit_pos[:n_groups] #pegando qnt certa de posições permitidas q é a mesma do numero de grupos
        #print('n grupos {}, dados agrupados {}, sobra no final {}'.format(n_groups,dados_agrupados,sobra))
        
        #add o que sobra no final na lista de posiçoes permitidas
        if sobra > 0:
            if not contrast_final: #usamos isso pra destacar o final que foi adicionado
                permit_pos.extend(all_positions[-sobra:])
            else:
                permit_pos.append(all_positions[-sobra:])
    
        return permit_pos
    
    def __findPos_filttered(self, original_pos, permitted_pos):
        """
        Return the positions that we should put 1 on the
        fingerprint already considering the positions
        allowed by the degree of freedom.
        """
        original_pos = np.asarray(original_pos)
        permitted_pos = np.asarray(permitted_pos)
        
        posINpermittedArray = []
        for i in range(0, len(original_pos)):
            posINpermittedArray.append(abs(original_pos[i] -  permitted_pos).argmin())    
        
        return permitted_pos[posINpermittedArray]

    def __forbiddenPos(self, permitted_pos):
        """
        Returns the forbidden positions in fingerprint array.
        """
        permitted_pos = np.asarray(permitted_pos)

        forbidden_pos = []
        for i in range(0, permitted_pos[-1]):
            if i not in permitted_pos:
                forbidden_pos.append(i)
        return forbidden_pos

    def gen_nmrFP(self, sample, degree_freedom = 1, spurious_variables=False, precision=1):
        """
        This function returns a binary vector of 0 and 1,
        where 1 means the presence of a signal in a given region.
        EX: ppm = [0.0, 0.3, 0.5], considering a vector from
        0 to 0.5 our fingerprint is [1, 0, 0, 1, 0, 1],
        where 0.0 corresponds to the first position of the
        vector and 0.5 the last position.

        Arguments
        ------------------------------------------
        sample
            A list of signals.
        
        degree_freedom
            the degree of freedom is associated with the
            rigidity of the system. Given a degree of
            freedom of 1, we have that in a vector of
            signals [0.0, 0.1, 0.2] only the value of the
            medium will be considered, i.e., 0.2 and 0.0 
            will be treated as 0.1. This idea arose to try 
            to deal with the fluctuations of signals that
            happen depending on the chemical environment.
            In this case, the greater the degree of freedom,
            the more rigid the system becomes.
        
        spurious_variables
            Spurious variables are positions in the vector
            that will always have a value of 0, set False
            to exclude them and True to maintain.

        precision
            Is the precision of the spectroscopic measure,
            equal to that reported in the range_spectra 
            variable (class variable).
        """
        axis_ = [self.start, self.stop, self.step]
        sample = np.asarray(sample)
        original_pos = SpectraFP.__findPos(self, sample, dec=precision) #get positions without degree of freedom
        permitted_pos = SpectraFP.__permittedPos(self, degree_freedom=degree_freedom, axis_range=axis_) #get positions with degree_Freedom

        new_positions = SpectraFP.__findPos_filttered(self, original_pos, permitted_pos)

        nmrfp_ = np.zeros(len(self._vet_fp))
        nmrfp_[new_positions] = 1
        if spurious_variables:
            return nmrfp_
        else:
            #get all forbidden positions in fingerprint array
            forbidden_pos = SpectraFP.__forbiddenPos(self, permitted_pos=permitted_pos)
            
            self.__x_axis_permitted_ = np.delete(self.__x_axis_permitted_, forbidden_pos)#get out xi forbiddens
            self.x_axis_permitted_ = self.__x_axis_permitted_.copy() #in case of multiple FP generation
            self.__x_axis_permitted_ = self._vet_fp.copy() #reset this variable to the normal
            
            nmrfp_ = np.delete(nmrfp_, forbidden_pos)
            return nmrfp_