# SpinCo
## Spindle detection application for Cognition database

### Required steps:
1- clone repository to [your folder]

2- create spinco.env in the repository folder file like:

    DATAPATH='[the path to COGNITION database in your computer]'
    
(spinco.env is a environment file -IT IS NEVER SHARED THROUGH GIT, see .gitignore- it serves to define variables specific of each machine and/or user)

3- add [your folder]/src to the path environment variable of your system manually (in order to use import of the library)

### Demo:
1- Download publicly available DREAMS database from "Stephanie Devuyst. (2005). The DREAMS Databases and Assessment Algorithm [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2650142"

2- Extract the .rar content to [your folder]/demo/DREAMS

3- Run the notebooks in order

### Cite:
If you use this code kindly cite:
"Beyond the ground truth. XGBoost by-sample model applied to the problem of sleep spindle event detection. E. Gurdiel, J. Gómez-Pilar and Roberto Hornero. (To Be Published)"

### Disclaimer:
This is research code, is distributed with no guaranty. No model is shared in this repository. This repository is not designed to be used in a production environment. The intention is to facilitate the actual code used in the research, not a ready-to-use product.

For any doubt regarding funcionality contact quiquegurdiel (at) gmail.com.

You should manage the dependencies by yourself, pip install [library name] should be enough for all libraries, check the imports at the begining of the notebooks.

Using virtual environmets is higly recommended.

### Structure of the repository:
Under SpinCo/src there is file named spinco.py that is the library used in the notebooks. The file is separated in different sections to facilitate the search for specific functions, in particular:
- Helpers
- Mathematics
- Filtering
- Band statistic
- Time domain input
- Frequency domain input
- Time-window metrics
- Spindle Detection 1
- Database management
- Feature & label management
- Experiment & model management
- Metrics
We recomend to run the demo and analise the functions that are called in order to understand the structure of the actual research notebooks, reading straight from the library might be less intuitive and could en up in reading code that is not actually used on the part of interest of the reader.

### Feature extraction:
The notebooks used for feature extraction are right under Spinco and use the prefixes "Extraction001_" to "Extraction007_".

### Feature selection:
Feature selection notebook is right under SpinCo in the file "Experiment008_featureSelection_DREAMS.ipynb".

### Experiments:
Under SpinCo/experiments there are several experiments, the results published are computed in the following experiments that have a subfolder associated:
- 010_COGNITION_testing26Features_7S_09aa67d8-c865-4a95-a7c5-de7b6adadbce
- 011_MASS_26Features_final_5388ca14-a315-4598-97c2-d44175b24937
