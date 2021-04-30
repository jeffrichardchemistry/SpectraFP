from setuptools import setup, find_packages

with open("README.md", 'r') as fr:
	description = fr.read()

setup(
    name='SpectraFP',
    version='1.0.0',
    url='https://github.com/jeffrichardchemistry/SpectraFP',
    license='GNU GPL',
    author='Jefferson Richard',
    author_email='jrichardquimica@gmail.com',
    keywords='Cheminformatics, Chemistry, QSAR, QSPR, Fingerprint, Spectroscopy',
    description='A package to perform fingerprints from spectroscopy datas.',
    long_description = description,
    long_description_content_type = "text/markdown",
    packages=['SpectraFP'],
    install_requires=['pandas', 'numpy', 'tqdm'],
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
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',]
)
