from setuptools import setup, find_packages

with open("README.md", 'r') as fr:
	description = fr.read()

setup(
    name='SpectraFP',
    version='1.2.5',
    url='https://github.com/jeffrichardchemistry/SpectraFP',
    license='GNU GPL',
    author='Jefferson Richard',
    author_email='jrichardquimica@gmail.com',
    keywords='Cheminformatics, Chemistry, QSAR, QSPR, Fingerprint, Spectroscopy',
    description='A package to perform fingerprints from spectroscopy datas.',
    long_description = description,
    long_description_content_type = "text/markdown",
    #need this when we have more then 1 python file in same __init__.py
    packages=find_packages(include=['fastsimilarity.py', 'SpectraFP']),
    include_package_data=True,
    install_requires=['pandas<=2.0.3', 'numpy<=1.24.4', 'tqdm'],
    #install_requires=['pandas<=2.0.3', 'numpy<=1.24.4', 'tqdm', 'numba>=0.54.1'],
	classifiers = [
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Chemistry',
		'Topic :: Scientific/Engineering :: Physics',
		'Topic :: Scientific/Engineering :: Bio-Informatics',
		'Topic :: Scientific/Engineering',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Natural Language :: English',
		'Operating System :: Microsoft :: Windows',
		'Operating System :: POSIX :: Linux',
		'Environment :: MacOS X',
		'Programming Language :: Python :: 3.8',
		'Programming Language :: Python :: 3.9',
		'Programming Language :: Python :: 3.10']
)
