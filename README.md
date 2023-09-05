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
This is research code, is distributed with no guaranty.

For any doubt regarding funcionality contact quiquegurdiel (at) gmail.com.

You should manage the dependencies by yourself, pip install [library name] should be enough for all libraries, check the imports at the begining of the notebooks.

Using virtual environmets is higly recommended (always).
