[![DOI](https://zenodo.org/badge/363234585.svg)](https://zenodo.org/badge/latestdoi/363234585) [![PyPI version](https://img.shields.io/pypi/v/spectrafp)](https://pypi.python.org/pypi/spectrafp) [![PyPI Downloads](https://pepy.tech/badge/spectrafp)](https://pepy.tech/project/spectrafp)

# Publication

This is the official repository for the article entitled: "SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications" published in the journal Physical Chemistry Chemical Physics (PCCP).

DOI: [https://doi.org/10.1039/D3CP00734K](https://doi.org/10.1039/D3CP00734K)

### How to cite
**Bibtex**:
```
@article{dias2023spectrafp,
  title={SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications},
  author={Dias-Silva, Jefferson R and Oliveira, Vitor M and Sanches-Neto, Fl{\'a}vio O and Wilhelms, Renan Z and J{\'u}nior, Luiz HK Queiroz},
  journal={Physical Chemistry Chemical Physics},
  volume={25},
  number={27},
  pages={18038--18047},
  year={2023},
  publisher={Royal Society of Chemistry}
}
```
**Text**:
```
Dias-Silva, Jefferson R., et al. "SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications." Physical Chemistry Chemical Physics 25.27 (2023): 18038-18047.
```
```
Dias-Silva, J. R., Oliveira, V. M., Sanches-Neto, F. O., Wilhelms, R. Z., & Júnior, L. H. Q. (2023). SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications. Physical Chemistry Chemical Physics, 25(27), 18038-18047.
```
```
DIAS-SILVA, Jefferson R., et al. SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications. Physical Chemistry Chemical Physics, 2023, 25.27: 18038-18047.
``` 

# SpectraFP
SpectraFP is a package to perform fingerprints from spectroscopy datas. The goal is to transform a list of spectroscopy signals - such as nmr and infrared - into a binary vector of zeros and ones. One means that there is a sign and zero absence.

## Install
<b>Via pypi</b>
```
$ pip install spectrafp
```

<b>Via github</b>
```
$ git clone https://github.com/jeffrichardchemistry/SpectraFP
$ cd SpectraFP
$ python3 setup.py install
```

# Tutorials
Tutorials on how to use this package as well as loading the databases are available in the jupyter-notebook file in `example/how_to_use.ipynb` and `Reading pickle data.ipynb`respectively.
