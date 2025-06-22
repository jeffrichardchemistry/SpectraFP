import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import os
from parallel_pandas import ParallelPandas


try:
    from .fastsimilarity import getOnematch #to package
except ImportError:
    from fastsimilarity import getOnematch #run locally

ABSOLUT_PATH = os.path.dirname(os.path.realpath(__file__))
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

    def genFP(self, sample, correction = 1, spurious_variables=False, precision=1):
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
        
        correction
            the correction is associated with the
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
        permitted_pos = SpectraFP.__permittedPos(self, degree_freedom=correction, axis_range=axis_) #get positions with degree_Freedom

        new_positions = SpectraFP.__findPos_filttered(self, original_pos, permitted_pos)

        nmrfp_ = np.zeros(len(self._vet_fp), dtype='uint8')
        nmrfp_[new_positions] = np.uint8(1)
        if spurious_variables:
            self.x_axis_permitted_ = self.__x_axis_permitted_.copy()
            return nmrfp_
        else:
            #get all forbidden positions in fingerprint array
            forbidden_pos = SpectraFP.__forbiddenPos(self, permitted_pos=permitted_pos)
            
            self.__x_axis_permitted_ = np.delete(self.__x_axis_permitted_, forbidden_pos)#get out xi forbiddens
            self.x_axis_permitted_ = self.__x_axis_permitted_.copy() #in case of multiple FP generation
            self.__x_axis_permitted_ = self._vet_fp.copy() #reset this variable to the normal
            
            nmrfp_ = np.delete(nmrfp_, forbidden_pos)
            return nmrfp_

    
    def fit(self, data_signs, correction=1, precision=1, spurious_variables=False, returnAsDataframe=True, colabel='sign-',verbose=True):
        """
        Generate fingerprints of a signs list.
        
        Args:
            data_signs:
                List or matrix of signs.
                
                >>> Example of data format
                array([[1.2, 2.4, 12.3, 145.0],
                       [5.2, 176.3],
                       [12.4, 30.1, 50.3, 70.4,188.7],
                       ...,
                       [2.2, 65.4, 76.1,125.3]])
                    
            correction:
                The correction is associated with the
                rigidity of the system. Given a degree of
                freedom of 1, we have that in a vector of
                signals [0.0, 0.1, 0.2] only the value of the
                medium will be considered, i.e., 0.2 and 0.0 
                will be treated as 0.1. This idea arose to try 
                to deal with the fluctuations of signals that
                happen depending on the chemical environment.
                In this case, the greater the degree of freedom,
                the more rigid the system becomes.
                
            precision:
                Is the precision of the spectroscopic measure,
                equal to that reported in the range_spectra 
                variable (class variable).
                
            spurious_variables:
                Spurious variables are positions in the vector
                that will always have a value of 0, set False
                to exclude them and True to maintain.
                
            returnAsDataframe:
                Return fingerprints in Pandas dataframe format.
            colabel:
                Label of column name in dataframes. Use when
                'returnAsDataframe = True'
            verbose:
                Show the progressbar.
            
        Returns:
            returns a dataframe or matrix of fingerprints
        
        """
        fps = []
        if verbose:
            for sign in tqdm(data_signs):
                fps.append(self.genFP(sample=sign,
                                      correction=correction,
                                      spurious_variables=spurious_variables, 
                                      precision=precision))
        else:
            for sign in data_signs:
                fps.append(self.genFP(sample=sign,
                                      correction=correction,
                                      spurious_variables=spurious_variables, 
                                      precision=precision))
        
        fps = np.array(fps)
        if returnAsDataframe:
            np.set_printoptions(suppress=True)
            df_fps = pd.DataFrame(fps, columns=[colabel+str(round(i,precision)) for i in self.x_axis_permitted_])
            
            return df_fps
        else:
            return fps

class SpectraFP1H():
    def __init__(self, range_spectra: list = [0,10,0.01], multiplicty_filter:list = ['All']):
        self.range_spectra = range_spectra
        self.multiplicity_filter = multiplicty_filter
        
        if multiplicty_filter != ['All']:
            self.multiplicity_filter = [string.lower() for string in self.multiplicity_filter]
            self.multiplicity = self.__filterMultiplicityDF(dfmultiplicity=self.__multiplicityList())            
        else:
            self.multiplicity = self.__multiplicityList()
        self.allppms = np.arange(self.range_spectra[0],self.range_spectra[1],self.range_spectra[2]).astype(float).round(2)
    
    def genFP(self,peaks: list, correction: int = 0 , returnAsBinaryValues: bool = False):
        self.__checkIfValGreaterScale(peaks) # checks if there are values in peaklist greater then upper scale limit   

        self.__checkMultiplicityFilterInInputData(df_filtered=self.multiplicity,data=peaks) #checks if multiplicities of input data is in multiplicity df filtered
        
        indexes = self.__getIndexes(peaks=peaks) #get indexes of ppm, multiplicty and number of hydrogen when possible
        matrixfp = self.__makeCorrelationMatrix(indexes=indexes) #built fingerprint matrix
        if correction > 0:
            matrixfp = self.__compressFP(fingerprintArray=matrixfp, correction=correction)

        if returnAsBinaryValues:
            matrixfp[matrixfp>1] = 1
        return matrixfp   

    def __compressFP(self, fingerprintArray:np.ndarray ,correction:int=1):
        n = correction #fator de correção
        grid = 2*n+1 #1, 3, 5, 7, 9 .... #grid para concatenar

        ar = fingerprintArray.copy()

        tile = ar.shape[1] // grid # numero de grids completo que vamos ter
        rest = ar.shape[1] % grid #colunas que sobram no final

        fpstacked = np.empty((ar.shape[0],0),int) #matriz vazia para juntar as colunas concatenadas
        for i in range(tile): #Iteração para cada grid completo
            #coluna individual contatenada-somada
            gridstacked_column = ar[:,i*grid:grid*(i+1)].sum(axis=1).reshape(-1,1)
            #Juntando todas as colunas contatenada-somada na mesma matriz
            fpstacked = np.hstack((fpstacked,gridstacked_column))

        if rest != 0: #em casos de resto 0 não sobra colunas a ser concatenada na matriz final. 
            rest_finalfp = ar[:,-rest:] #pega o resto das colunas que sobra e concatena na matriz final
            fpstacked = np.hstack((fpstacked,rest_finalfp))

        return fpstacked

    def __makeCorrelationMatrix(self, indexes: list):
        """Checks the shape of data and built the matrix based on shape (n, 3) "case that has number of hydrogens" and
        shape (n, 2) "doesn't have number of hydrogen".

        Args:
            indexes (list): _description_

        Returns:
            _type_: _description_
        """
        ar = np.array(indexes).shape
        if ar[1] == 3:
            matrix = np.zeros((self.multiplicity.shape[0], len(self.allppms)))
            for indx in indexes:
                matrix[indx[1],indx[0]] = indx[2]
            return matrix
        
        elif ar[1] == 2:
            matrix = np.zeros((self.multiplicity.shape[0], len(self.allppms)))
            for indx in indexes:
                matrix[indx[1],indx[0]] = 1
            return matrix
    
    def __getIndexes(self, peaks: list):
        """Search for the indexes of ppms and multiplicities and returns a list of tuples where:
        [(index_ppm, index_multiplicity, number of hydrogen)]

        Args:
            peaks (list): _description_

        Returns:
            _type_: _description_
        """    
        ar = np.array(peaks).shape
        try:
            t = ar[1]
        except IndexError as e: #test shape of the list and print the error
            raise IndexError("Data structure must be a list of tuples with the shape equal (n,3) or (n,2). Example: [(1.23,'d',2), (3.21,'t',3)]") 
        indexesAndNum1H = []
        if ar[1] == 3:
            for peak_multi in peaks:
                index_ppm = np.where(self.allppms == peak_multi[0])[0][0] #search index of a ppm inside array
                index_multiplicity = self.__searchForMultiplicityInDF(peak_multi[1]) #search index of a multiplicity inside dataframe (search in both columns)
                indexesAndNum1H.append((index_ppm, index_multiplicity, peak_multi[2]))
                
            return indexesAndNum1H
        elif ar[1] == 2:
            for peak_multi in peaks:
                index_ppm = np.where(self.allppms == peak_multi[0])[0][0] #search index of a ppm inside array
                index_multiplicity = self.__searchForMultiplicityInDF(peak_multi[1]) #search index of a multiplicity inside dataframe (search in both columns)
                indexesAndNum1H.append((index_ppm, index_multiplicity))
                
            return indexesAndNum1H
        
        else:
            raise IndexError("Data structure must be a list of tuples with the shape equal (n,3) or (n,2). Example: [(1.23,'d',2), (3.21,'t',3)]")
    
    def __checkIfValGreaterScale(self,peaks):
        for info in peaks:
            if info[0] > self.range_spectra[1]:
                #print(f"There is a value greater then scaled in peaklist: {info[0]}")                
                raise InconsistentvalueError(f"There is a value greater then scale in peaklist. The limit of scale is {self.range_spectra[1]} and peaklist has {info[0]}")

    def __checkMultiplicityFilterInInputData(self,df_filtered: pd.DataFrame, data: list):
        """This function checks if there is a multiplicity in input data that is not in multiplicity dataframe filtered.

        Args:
            df_filtered (pd.DataFrame): _description_
            data (list): _description_
        """
        
        
        values_to_check = [dats[1] for dats in data]
        #print(values_to_check)
        acronym_set = set(df_filtered['acronym'].str.lower())
        name_set = set(df_filtered['name'].str.lower())
        missing_values = [value for value in values_to_check if value.lower() not in acronym_set and value.lower() not in name_set]

        if missing_values != []:
            raise MultiplicityError(f"The input data has multiplicities that are not in the list of filtered multiplicities. The following multiplicities are not contained: {set(missing_values)}")
        #print(missing_values)
            
    def __searchForMultiplicityInDF(self, multiplicity: str):
        """This function search the position of a multiplicity inside the pandas dataframe
        and returns the index. This function includes both type of multiplicity names and acronyms

        Args:
            multiplicity (str): _description_

        Returns:
            _type_: _description_
        """
        multip_col1 = self.multiplicity.loc[self.multiplicity['acronym'] == multiplicity.lower()]
        multip_col2 = self.multiplicity.loc[self.multiplicity['name'] == multiplicity.lower()]
        if multip_col1.empty and multip_col2.empty:
            multiplicity = 'm'
            return self.multiplicity.loc[self.multiplicity['acronym'] == multiplicity.lower()].index[0]
        elif multip_col1.empty:
            return multip_col2.index[0]
        else:
            return multip_col1.index[0]
    
    def __filterMultiplicityDF(self, dfmultiplicity:pd.DataFrame):
        df = dfmultiplicity.copy()
        df = df[(df['acronym'].isin(self.multiplicity_filter)) | (df['name'].isin(self.multiplicity_filter))]
        return df
            
    def __multiplicityList(self):
        path = ABSOLUT_PATH+'/data/multiplicities.csv'
        df = pd.read_csv(path,sep=',')
        df['acronym'] = df['acronym'].str.lower()
        df['name'] = df['name'].str.lower()
        return df
            
class SearchEngine:
    def __init__(self):
        pass
    
    def __loadDataSets(self,correction=3):
        """
        This function return a dataframe with spectraFP and smiles.

        Parameters
        ----------
        correction : TYPE, int
            DESCRIPTION. Database corrections to SpectraFP. The default is 3.

        Returns
        -------
        TYPE
            Pandas dataframe.

        """
        path = ABSOLUT_PATH+'/data'        
        filenames = os.listdir(path)
        for name in filenames:
            if correction > 4:
                print('Correction must be 0,1,2,3 or 4')
                return 0
            if str(correction) in name:
                return pd.read_pickle(path+'/'+name, compression='zip')
    
    def __makeSpectraFP(self, peaklist, correction = 3, spurious_variables=False, precision=1):
        """
        Transform a input of peaklist in SpectraFP.

        Parameters
        ----------
        peaklist : list
            list of signs.
        correction : int
            DESCRIPTION. The default is 3.
        spurious_variables : Bool
            DESCRIPTION. The default is False.
        precision : int
            DESCRIPTION. The default is 1.

        Returns
        -------
        sfp : array
            SpectraFP.

        """
        se = SpectraFP(range_spectra=[0, 240, 0.1])
        sfp = se.genFP(sample=peaklist, correction = correction, spurious_variables=False, precision=1)
        return sfp
    
    def __getMatch(self, threshold, difBetween13C,similarity_metric, alpha, beta,nsigns_input,onlySpecFP, input_FP, complete_base):
        _, matches_complete = getOnematch(threshold=threshold,
                                          base_train=onlySpecFP, base_test=input_FP, complete_base=complete_base,
                                          similarity_metric=similarity_metric,alpha=alpha, beta=beta)
        
        if not difBetween13C:
            return _
        
        else:
            new_matches = {}               
            for smi,sim_signs in matches_complete.items():
                dInput_match = int(abs(sim_signs[1]-nsigns_input))
                if dInput_match <= difBetween13C:
                    new_matches[smi] = sim_signs[0]
            
            return new_matches
    
    def search(self,signs_list=[],threshold=0.8,difBetween13C=5,correction=3,similarity='tanimoto', alpha=1, beta=0.5):
        """
        This function perform a similarity search between spectraFPs (query -> database) from a list of signs.

        Parameters
        ----------
        signs_list : TYPE, list
            List of signs, must be inside the interval
            0.0 to 240.0. example [0.1, 14.5, 25.3, 125.3, 190.3].
        threshold : TYPE, float
            Must be inside interval 0 to 1. The default is 0.8.
        difBetween13C : TYPE, int
            This parameter is a kind of filter. This means
            the difference between the amount of signs, in spectraFP form,
            between the query sample and the samples from database.
            The default is 5. Set 'False' to do not use this filter.
        correction : TYPE, int
            The correction is associated with the
            rigidity of the system. Given a degree of
            freedom of 1, we have that in a vector of
            signals [0.0, 0.1, 0.2] only the value of the
            medium will be considered, i.e., 0.2 and 0.0 
            will be treated as 0.1. This idea arose to try 
            to deal with the fluctuations of signals that
            happen depending on the chemical environment.
            In this case, the greater the degree of freedom,
            the more rigid the system becomes.
            The default is 3.
        similarity : TYPE, str
            The calculation metric to use. The options are:
                'tanimoto','tversky','geometric', 'arithmetic', 'euclidian','manhattan'
            
            The default is 'tanimoto'.
        alpha : TYPE, int/float
            Only use if similarity is tversky.
            The default is 1.
        beta : TYPE, int/float
            Only use if similarity is tversky.
            The default is 0.5.

        Returns
        -------
        matches_complete : TYPE, dictionary
            Dictionary with smiles as keys and similarities as values.

        """
        
        db = self.__loadDataSets(correction=correction) #load database with currently corretion
        input_specFP = self.__makeSpectraFP(peaklist=signs_list, correction = correction, spurious_variables=False, precision=1)
        
        matches_complete = self.__getMatch(threshold=threshold, difBetween13C=difBetween13C,
                                              similarity_metric=similarity, alpha=alpha, beta=beta,
                                              nsigns_input=input_specFP.sum(),
                                              onlySpecFP=db.iloc[:, 1:].values.astype('uint64'),
                                              input_FP=input_specFP.reshape(1,-1).astype('uint64'),
                                              complete_base=db)
        
        matches_complete = dict(sorted(matches_complete.items(), key=lambda x: x[1], reverse=True))
        
        
        return matches_complete

class SearchMetabolitesBy1H:

    def __init__(self) -> None:
        self.db = self.__loadata()

    def search(self,signs1H:list, correction:int=1, range_spectra:list=[0,10,0.01], threshold:float=0.6, difLenghtSigns:int=2 ,multiplicty_filter=['All'], n_threads:int=1):
        useintegral = self.__needIntegral(signs1H)
        spectrafpColumn2use:None
        if useintegral:
            spectrafpColumn2use = 'spectraFP_dtype'
        else:
            spectrafpColumn2use = 'spectraFP_NoIntegral'
        
        ParallelPandas.initialize(disable_pr_bar=True,n_cpu=n_threads) 
        df = self.db.copy()

        hfp_ = SpectraFP1H(range_spectra=range_spectra,multiplicty_filter=multiplicty_filter)
        signs1H_vet = hfp_.genFP(signs1H,correction=correction).ravel()

        df['DifSigns'] = df[spectrafpColumn2use].p_apply(lambda x: abs(len(x) - len(signs1H)))
        df = df.query('DifSigns <= @difLenghtSigns').reset_index(drop=True)
        #df.drop('DifSigns',axis=1,inplace=True)
        df['Similarity'] = df[spectrafpColumn2use].p_apply(lambda x: self.cosineSimilarity(hfp_.genFP(x,correction=correction).ravel(), signs1H_vet))
        df = df.query('Similarity >= @threshold')
        df = df.sort_values(by=['Similarity'], ascending=False).reset_index(drop=True)
        #print(df)
        return df

    def __needIntegral(self,sign1H):
        signshape = np.array(sign1H).shape
        if signshape[1] == 3:
            return True
        else:
            return False

    def __loadata(self):
        path = f'{ABSOLUT_PATH}/data/metabolites.pkl'        
        return pd.read_pickle(path,compression='zip')    
    
    def cosineSimilarity(self,vet1,vet2):
        return dot(vet1, vet2) / (norm(vet1) * norm(vet2))
    
    @staticmethod
    def drawSimilarMolecules(dfSimilarity1H:pd.DataFrame,n_molecules2show:int=10,path2save:str=None,filename:str=None):
        from rdkit.Chem import Draw
        from rdkit import Chem

        nfounded = dfSimilarity1H.shape[0]
        if n_molecules2show > nfounded:
            n_molecules2show = nfounded

        top_10_molecules = [Chem.MolFromInchi(dfSimilarity1H['Inchi'][i]) for i in range(0,n_molecules2show)]
        top_10_names = [dfSimilarity1H['Name'][i] for i in range(0,n_molecules2show)]
        top_10_similarity = [dfSimilarity1H['Similarity'][i] for i in range(0,n_molecules2show)]
        legend = [f'{name}\n{round(sim*100,2)}%' for name,sim in zip(top_10_names,top_10_similarity)]
        img=Draw.MolsToGridImage(top_10_molecules,molsPerRow=4,subImgSize=(300,300),legends=legend,useSVG=True)
        if path2save == None:
            return img
        else:
            if filename == None:
                with open(f'{path2save}/molsgridSimilarity.svg', 'w') as file:
                    file.write(img)
            else:
                with open(f'{path2save}/{filename}.svg', 'w') as file:
                    file.write(img)
            

## classes de excepts personalizados
class MultiplicityError(Exception):
    pass

class InconsistentvalueError(Exception):
    pass

if __name__ == '__main__':
    ################# Testes SpectraFP    
    #amostra = [0.0, 12.4,0.1, 25.4,25.5,25.6, 35.1, 70.4, 170.4, 175.2, 187]
    #nfp = SpectraFP(range_spectra=[0, 187, 0.1])    
    #amostra2 = [0.1, 0.2, 0.5, 0.8, 2.1, 2.9]
    #amon = [[0.1, 0.4, 17, 110, 180], [3, 5, 10,12,16, 150], [43, 78, 170]]
    #nfp = SpectraFP(range_spectra=[0,190,0.1])
    #get = nfp.fit(data_signs=amon, correction=2, spurious_variables=False, returnAsDataframe=False)
    #nfp = SpectraFP(range_spectra=[0, 3, 0.1])
    #get = nfp.genFP(sample=amostra2, degree_freedom=2, spurious_variables=True)
    #print(get, len(get), nfp.x_axis_permitted_)
    
    ################# Testes SearchEngine
    #ppm_example = [8.1, 9.0, 13.5, 13.7, 18.2, 18.3, 104.6, 108.4, 109.4, 112.4, 113.2, 116.4, 120.9, 121.0, 137.4, 137.5, 145.6, 146.0, 151.2, 159.5, 159.8, 168.1, 171.7]
    #ppm_example2 = [13.4, 14.0, 22.7, 27.1, 29.4, 29.7, 29.8, 30.8, 32.0, 46.4, 204.4] #nice example
    #se = SearchEngine()
    #get = se.search(signs_list=ppm_example2, difBetween13C=False,similarity='geometric', threshold=0.2, correction=0)
    #print(get)

    ################# Testes SpectraFP1H
    data = [(7.74, 'd', 1), (7.5, 'd', 1), (7.23, 'm', 2), (7.16, 'td', 1), (4.33, 'dd', 1), (3.25, 'm', 2), (3.06, 'dd', 2)]
    data = [(7.74, 'd', 1), (7.5, 'd', 1), (7.23, 'm', 2), (7.16, 'td', 1), (4.33, 'dd', 1), (3.25, 'm', 2), (3.06, 'dd', 2)]
    #data = [(0.03,'multiplet',2),(1.12,'s',4),(2.50,'s',3),(0.18,'q',1),(0.12,'t',2)]
    #data = [(4.3, 'm', 2), (3.9, 'm', 3), (3.64, 'm', 4), (3.2, 's', 9)]
    data = [(11.96, 'd', 1), (11.57, 'd', 1), (11.5, 's', 1), (7.91, 'm', 1), (6.99, 's', 3), (5.92, 'd', 6)]
    data = [(7.69, 'd', 1), (7.64, 'dd', 1), (6.91, 'd', 1), (2.23, 's', 3)]
    data = [(7.69, 'd'), (7.64, 'dd'), (6.91, 'd'), (2.23, 's')]
    data = [(11.96, 'd'), (11.57, 'd'), (11.5, 's'), (7.91, 'm'), (6.99, 's'), (5.92, 'd')]
    data = [(7.08, 's', 1), (6.95, 's', 2), (3.89, 's', 3), (3.25, 'm', 2)]

    hfp = SpectraFP1H(range_spectra=[0,14,0.01],multiplicty_filter=['All'])
    result = hfp.genFP(peaks=data, correction=2, returnAsBinaryValues=False)
    #print(result,result.shape)

    ##teste search engine metabolites
    sem = SearchMetabolitesBy1H()
    db_sim = sem.search(signs1H=data,
               correction=3,
               threshold=0.2,
               difLenghtSigns=2,
               range_spectra=[0,14,0.01],
               multiplicty_filter=['All'],
               n_threads=6)
    print(db_sim)
    
    SearchMetabolitesBy1H.drawSimilarMolecules(dfSimilarity1H=db_sim,
                                               n_molecules2show=10,
                                               path2save='/home/jefferson',
                                               filename=None)
























