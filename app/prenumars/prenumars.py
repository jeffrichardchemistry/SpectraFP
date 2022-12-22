import builtins
from copyreg import pickle
import streamlit as st
from streamlit_option_menu import option_menu #pip3 install streamlit-option-menu
import numpy as np
from PIL import Image
import base64
import pickle
import os
import pandas as pd
from stqdm import stqdm
from pyADA import ApplicabilityDomain
import xgboost

from rdkit.Chem import AllChem, Draw
from rdkit import Chem
from rdkit.Chem import rdDepictor, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

from texts import Texts
from fastsimilarity import getOnematch
from specFP import SpectraFP

class BackEnd:
    def __init__(self):
        self.start = 0
        self.stop = 240
        self.step = 0.1
        self.vet_fp = np.arange(self.start, self.stop, self.step)

        #pyADA
        self.ad = ApplicabilityDomain()

        #base train
        self.bases_train = BackEnd._load_basetrain(self)
        
        #models
        self.models = ''
        
    #@st.cache(allow_output_mutation=True)
    def _load_basetrain(self):
        path = 'dataAD/'
        #path = 'dataAD\'
        filenames = [name.replace('.pkl', '') for name in os.listdir(path)]
        pathnames = [path+name for name in os.listdir(path)]
        #store all df in a dictionary
        allfiles = {}
        for names, paths in zip(filenames, pathnames):
            allfiles[names] = pd.read_pickle(paths, compression='zip')
            
        return allfiles

    @st.cache(allow_output_mutation=True) #need this to cache models
    def _loadmodels(self):
        path = 'models/'
        path_fragsDIR = [path+path_frags for path_frags in os.listdir(path)]
        filenames = [path_frags for path_frags in os.listdir(path)] #filenames is folders name (without .pkl)
        
        models_path = {model_name: ['{}{}/{}'.format(path,model_name,x) for x in os.listdir(model_path)] for model_name, model_path in zip(filenames, path_fragsDIR)}
        
        models = {}
        for model_name, model_path in models_path.items():
            models[model_name] = [pickle.load(open(model, 'rb')) for model in model_path]
        
        return models
        
    def __moltosvg(self, mol, molSize = (320,320), kekulize = True):
        mol = Chem.MolFromSmiles(mol)
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace('svg:','')
       
    def _render_svg(self, path_img):
        f = open(path_img, 'r')
        svg = f.read()
        f.close()
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        return html

    def _applicabilityDomain(self, btrain, btest, th):
        get_simdf = self.ad.analyze_similarity(base_test=btest, base_train=btrain)
        similiraty = get_simdf['Max'].values[0]
        if similiraty >= th:
            return True, similiraty
        else:
            return False, similiraty    
    
    def load_baseSimilarity(self, degree_freedom):
        path_similarity = ['DB_similarity/'+names for names in os.listdir('DB_similarity/')]
        if degree_freedom == 0:
            db = [x for x in path_similarity if 'nmrFP_DF0_P' in x]
            df_full = pd.DataFrame()
            for files in stqdm(db, desc='Loading databases'):
                readed = pd.read_csv(files)
                df_full = pd.concat([df_full, readed], axis=0)            
            return df_full

        elif degree_freedom == 1:
            db = [x for x in path_similarity if 'nmrFP_DF1.csv' in x]
            df_full = pd.DataFrame()
            for files in stqdm(db, desc='Loading databases'): #in case of multiple files, and to show progbar
                readed = pd.read_csv(files)
                df_full = pd.concat([df_full, readed], axis=0)
            return df_full

        elif degree_freedom == 2:
            db = [x for x in path_similarity if 'nmrFP_DF2.csv' in x]
            df_full = pd.DataFrame()
            for files in stqdm(db, desc='Loading databases'): #in case of multiple files, and to show progbar
                readed = pd.read_csv(files)
                df_full = pd.concat([df_full, readed], axis=0)
            return df_full

        elif degree_freedom == 3:
            db = [x for x in path_similarity if 'nmrFP_DF3.csv' in x]
            df_full = pd.DataFrame()
            for files in stqdm(db, desc='Loading databases'): #in case of multiple files, and to show progbar
                readed = pd.read_csv(files)
                df_full = pd.concat([df_full, readed], axis=0)
            return df_full

        elif degree_freedom == 4:
            db = [x for x in path_similarity if 'nmrFP_DF4.csv' in x]
            df_full = pd.DataFrame()
            for files in stqdm(db, desc='Loading databases'): #in case of multiple files, and to show progbar
                readed = pd.read_csv(files)
                df_full = pd.concat([df_full, readed], axis=0)
            return df_full
    
    def plot_molecule(self, smiles, imgsize=(320,320)):
        svg = BackEnd.__moltosvg(self, mol=smiles, molSize=imgsize)
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64

        return html
        #st.write(html, unsafe_allow_html=True)    
    
    def makeFP(self, signs_list,df=3,spurious_variables = False):
        """This function transform a list of signs in array of 0's and
        1's (Fingerprints).

        Args:
            signs_list (list): list of signs
            df (int, optional): System rigidity. Defaults to 3.
            spurious_variables (bool, optional): Cut off variables that always be 0. Defaults to False.

        Returns:
            array: array of fingerprint
        """
        list_signs = np.array(signs_list)
        signFP = SpectraFP(range_spectra=[0, 240, 0.1]).gen_nmrFP(sample=list_signs, degree_freedom=df, spurious_variables=spurious_variables)
        return signFP

    def Komitee(self, fp):
        """This function make prediction of Functional groups by
        decision of 5 differents models.

        Args:
            fp (array): fingerprint arrays
            
        Returns:
            Retorna de label of a fragment related to the singnals of spectra
        """
        decision1 = {frag: np.array([model.predict(fp) for model in models]).ravel() for frag, models in self.models.items()}
        
        final_decision = []
        for frag, decisions in decision1.items():
            votes = sum(decisions)
            if votes >= 3:
                final_decision.append(frag)
        
        #THis block of code apply AD calculations to the resulted fragments by komitee        
        get_decision_AD = {} #get if the FP is in AD and the max similarity (to the future maybe)
        final_decision_W_AD = [] #get final decision with AD applied
        for frag_dec in final_decision: #iter over frags labels
            base_train = self.bases_train[frag_dec].values #getting train datasets
            ad_decision = BackEnd._applicabilityDomain(self, btrain=base_train, btest=fp, th=0.15)
            
            get_decision_AD[frag_dec] = list(ad_decision)
            if ad_decision[0]:
                final_decision_W_AD.append(frag_dec)
            
        return final_decision_W_AD
            
    def dirImgs(self, list_FG):
        path = 'figs/'
        path_file = [path+names for names in os.listdir(path)]
        keys = [names.replace('.png', '') for names in os.listdir(path)]
        imgs = {key: path for key, path in zip(keys, path_file)}
        
        filter_imgs = {}
        for fgs in list_FG:
            filter_imgs[fgs] = imgs[fgs]
        return filter_imgs

    def filterByNsigns(self, matches, nsigns_input,dif_nsigns):
        """This function return a dictionary of smiles as keys
        and similarities as values and the filter already has been
        applied

        Args:
            matches (dictionary): dictionary of smiles, similarity and number of signs
            nsigns_input (int): signs number of spectraFP input
            dif_nsigns (int): signal difference between input spectrafp and database spectrafp

        Returns:
            dictionary: dictionary of smiles and similarity
        """
        matchs = {}
        for smi,sim_signs in matches.items():            
            dInput_match = int(abs(sim_signs[1]-nsigns_input))
            if dInput_match <= dif_nsigns:
                matchs[smi] = sim_signs[0]            
                
            #print(smi, sim_signs, dInput_match)
        return matchs
        
class FrontEnd(BackEnd):
    def __init__(self):
        super().__init__()
        gettext = Texts()
        self.infotext = gettext.info()
        self.text1 = gettext.text1()
        self.text2 = gettext.text2()
        self.text3 = gettext.text3()
        self.aboutext = gettext.about()
        
        FrontEnd.main(self)
    def NavigationBar(self):
        #st.sidebar.markdown('# Navegation:')
        #nav = st.sidebar.radio('Go to:', ['HOME', 'NMR Elucidator', 'Search Structures' ,'About'])        
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                menu_icon="cast",
                options = ["Home", "NMR Elucidator","Search Structures","About"],
                icons = ["house-door-fill","diagram-3-fill","search","info-circle-fill"], #Here we get bootstrap icons that are included in streamlit_options package (we need just write the name https://icons.getbootstrap.com/)
            )
        st.sidebar.markdown("# Contribute")
        st.sidebar.info('{}'.format(self.infotext))
        return selected
        
        #return nav

    def main(self):
        nav = FrontEnd.NavigationBar(self)
        if nav == 'Home':
            st.title('PreNuMaRS')            
            st.markdown('{}'.format(self.text1), unsafe_allow_html=True)        
            img1 = Image.open('figs_text/figura_SpecFP.png')
            left,mid,right = st.columns(3)
            mid.image(img1,width=300,caption='Figure 1. Schematic illustrating the creation of the spectraFP vector from spectrum data.')
            
            st.markdown('{}'.format(self.text2), unsafe_allow_html=True) 
            img2 = Image.open('figs_text/grupos_funcionais.png')
            st.image(img2,caption='Figure 2. Possible predicted functional groups.')
            
            st.markdown('{}'.format(self.text3), unsafe_allow_html=True)
            img3 = Image.open('figs_text/search_algorithm.png')
            left,mid2,right = st.columns(3)
            mid2.image(img3,width=300,caption='Figure 3. Workflow for using the search algorithm.')

        if nav == 'NMR Elucidator':           
            st.title('Functional Group Predictor')
            spec_default = '8.1, 9.0, 13.5, 13.7, 18.2, 18.3, 104.6, 108.4, 109.4, 112.4, 113.2, 116.4, 120.9, 121.0, 137.4, 137.5, 145.6, 146.0, 151.2, 159.5, 159.8, 168.1, 171.7'
            ppm = st.text_input('Type a nmr 13C peak list:', spec_default)
            ppm = [float(x) for x in ppm.split(',')] #transform str in list

            btn_predictor = st.button('Predict')
            if btn_predictor:
                self.models = BackEnd._loadmodels(self)#load models, this isn't cached yet
                
                fp = FrontEnd.makeFP(self, signs_list=ppm, df=3).reshape(1, -1)
                
                get_preds = FrontEnd.Komitee(self, fp=fp)
                if get_preds == []:
                    st.markdown('## No Functional groups found.')
                
                imgs2show = FrontEnd.dirImgs(self, list_FG=get_preds)
                for fg, path in imgs2show.items():
                    img = Image.open(path)
                    st.image(img)                            
                
        if nav == 'Search Structures':
            st.title('Search Structures')
            
            default_ppm = '14.2, 14.4, 22.9, 25.2, 29.4, 29.6, 29.8, 29.9, 32.2, 34.4, 60.1, 173.5'
            ppm = st.text_input('Type nmr carbon ppm', default_ppm)
            ppm = [float(x) for x in ppm.split(',')] #transform str in list
            
            col1__, col2__ = st.columns(2)
            search_th = col1__.slider('Type a minimum of similarity', 0.0, 1.0, 0.6)
            search_nsigns = col2__.slider('Choose a number to variation signs.', 0, 15, 5)
            
            col2, col3 = st.columns(2)
            metric = col2.selectbox("Choose a Similarity methods", ("Tanimoto", "Tversky", "Geometric", "Arithmetic", "Euclidian"), index=2) #Manhattan doesnt works well
            degree_freedom = col3.selectbox("Correction", (0, 1, 2, 3, 4), index=3)
            if "Tversky" in metric:
                col1_, col2_, col3_, col4_, col5_, col6_ = st.columns(6)
                alpha_ = float(col1_.text_input('Type alpha (Tversky)', 1))
                beta_ = float(col2_.text_input('Type beta (Tversky)', 1))
            elif metric == "Euclidian" or metric == "Manhattan":
                _, grid_filter, _ = st.columns(3)
                filter_molecules = grid_filter.radio('Select a number of molecule to display:', (5, 10, 15, 20))
            
            #Run ALL
            btn_search = st.button('Search')
            if btn_search:
                #Instancy spectraFP and generate FP
                fpnmr = SpectraFP(range_spectra=[0, 240, 0.1])
                input_FP = fpnmr.gen_nmrFP(ppm, degree_freedom=degree_freedom, spurious_variables=False)
                input_FP = input_FP.reshape(1, -1).astype('uint64')
                nsigns_input = input_FP.sum()
                
                #variables
                th_sim = float(search_th)
                method_sim = metric.lower() #need lower to set directly in function
                #database
                df_full = FrontEnd.load_baseSimilarity(self, degree_freedom)
                db = df_full.iloc[:, 1:].values.astype('uint64')
                
                #runing
                if metric == "Tversky":
                    _,matches_complete = getOnematch(threshold=th_sim, base_train=db, base_test=input_FP, complete_base=df_full, similarity_metric=method_sim,alpha=alpha_, beta=beta_)
                    matchs = FrontEnd.filterByNsigns(self, matches=matches_complete, nsigns_input=nsigns_input, dif_nsigns=search_nsigns) #filttering of differences signs
                    matchs = dict(sorted(matchs.items(), key=lambda x: x[1], reverse=True)) #sort by values

                    if matchs == {}:
                        st.markdown('## <font color="orange">All spectra have similarity less than {}%.</font>'.format(th_sim*100),unsafe_allow_html=True)
                    else:
                        col1, col2, col3 = st.columns(3)
                        for smiles, sims_ in matchs.items():
                            col2.write(FrontEnd.plot_molecule(self, smiles=smiles, imgsize=(200,200)),unsafe_allow_html=True)
                            col2.markdown('# <center>{:.2f}%</center>'.format(sims_*100), unsafe_allow_html=True)
                
                elif metric == "Euclidian" or metric == "Manhattan":
                    _,matches_complete = getOnematch(threshold=th_sim, base_train=db, base_test=input_FP, complete_base=df_full, similarity_metric=method_sim)
                    matchs = FrontEnd.filterByNsigns(self, matches=matches_complete, nsigns_input=nsigns_input, dif_nsigns=search_nsigns) #filttering of differences signs
                    matchs = dict(sorted(matchs.items(), key=lambda x: x[1], reverse=True)) #sort by values
                    matchs = dict(tuple(matchs.items())[0:filter_molecules])

                    col1, col2, col3 = st.columns(3)
                    for smiles, sims_ in matchs.items():
                        col2.write(FrontEnd.plot_molecule(self, smiles=smiles, imgsize=(200,200)),unsafe_allow_html=True)
                        col2.markdown('# <center>{:.2f}%</center>'.format(sims_*100), unsafe_allow_html=True)
                
                else:
                    #Get all matched structures
                    _,matches_complete = getOnematch(threshold=th_sim, base_train=db, base_test=input_FP, complete_base=df_full, similarity_metric=method_sim)
                    matchs = FrontEnd.filterByNsigns(self, matches=matches_complete, nsigns_input=nsigns_input, dif_nsigns=search_nsigns) #filttering of differences signs
                    matchs = dict(sorted(matchs.items(), key=lambda x: x[1], reverse=True)) #sort by values
                    
                                      
                    #identify empty matches
                    if matchs == {}:
                        st.markdown('## <font color="orange">All spectra have similarity less than {}%.</font>'.format(th_sim*100),unsafe_allow_html=True)              

                    else:
                        col1, col2, col3 = st.columns(3)
                    
                        for smiles, sims_ in matchs.items():
                            col2.write(FrontEnd.plot_molecule(self, smiles=smiles, imgsize=(200,200)),unsafe_allow_html=True)
                            col2.markdown('# <center>{:.2f}%</center>'.format(sims_*100), unsafe_allow_html=True)

        if nav == 'About':
            st.title('About')
            st.markdown('{}'.format(self.aboutext), unsafe_allow_html=True) 
            
            st.title('Supporters')
            img = Image.open('figs_text/supporters.png')
            st.image(img)
            
if __name__ == '__main__':
    run = FrontEnd()
