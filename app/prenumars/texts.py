from PIL import Image
import streamlit as st

class Texts:
    def __init__(self):
        pass

    def text1(self):        
        TEXT1 = """
        <body style='text-align: justify; color: black;'>
        <p>
        This system consists of applying the spectra-based descriptor (SpectraFP) in two case studies:
        (1) predicting six functional groups through machine learning (ML) models and (2) Searching for
        molecules in a database considering similarity between their spectra.
        The SpectraFP descriptor transforms a list of signals from a spectra into a standardized
        fingerprint of 0's and 1's. In addition, this descriptor is able to make small corrections in
        the variation of signals caused by external effects, pH, chemical environment, etc.
        </p>
        </body>             
        """        
        return TEXT1

    def text2(self):
        TEXT2 = """
        <body style='text-align: justify; color: black;'>
        <p>
        To predict the six functional groups shown in the figure below, 5 different ML models are used and
        the final decision is taken according to the equal results of most models, so if three or more
        models give the same result, then this is the final result.
        </p>
        </body> 
        """
        return TEXT2
    
    def text3(self):
        TEXT3 = """
        <body style='text-align: justify; color: black;'>
        <p>
        The Search Structures section conducts a search between a specified query spectra (peak list)
        and our spectras database. Here, you can select the similarity metric to be used as well as the amount 
        of signals that differ between the query spectra and the database spectra. When the similarity between
        the input and a sample from the database is equal to or greater than the preset threshold, the
        respective molecule from the database is output along with the similarity between the peaklists.
        </p>
        <p>
        The algorithm for obtaining SpectraFP and performing the similarity search can be used through the
        python package published on <a href="https://pypi.org/project/SpectraFP/">pypi</a> or installed manually by downloading the package
        <a href="https://github.com/jeffrichardchemistry/SpectraFP">here</a>.
        Tutorials on how to use these tools are accessible <a href="https://github.com/jeffrichardchemistry/SpectraFP/blob/main/example/how_to_use.ipynb">here</a>.
        </p>
        </body> 
        """
        return TEXT3

    def info(self):
        infotext = """
        This web app and SpectraFP package are free and open source and you are very welcome to contribute.
        This Application has GNU GPL license. All source code can be accessed [here](https://github.com/jeffrichardchemistry/SpectraFP).
        """
        return infotext
    
    def about(self):
        aboutext = """
        <body style='text-align: justify; color: black;'>
        <p>
        The SpectraFP descriptor was designed with the aim of introducing data from spectroscopic and spectrometric techniques in ML models to predict from
        physical-chemical properties to biological activities. Furthermore, this descriptor proved to be useful performing searches for molecular structures
        by comparing only their spectra. We hope these tools can help researchers in their fields, free of charge.
        </body>
        </p>
        """
        return aboutext
