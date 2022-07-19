# SpinCo
## Relation of spindles with cognitive variables using the COGNITION database

### Required steps:
1- clone repository

2- create spinco.env in the repository folder file like:

    DATAPATH='[the path to COGNITION database in your computer]'
    
(spinco.env is a environment file -IT IS NEVER SHARED THROUGH GIT, see .gitignore- it serves to define variables specific of each machine and/or user)

3- the database folder you are pointing to should include the csv with the results of the cognitive tests in the patients, contact enrique.gurdiel(at)uva.es if you have errors of missing .csv files related to it

4- have fun

### Disclaimer:
You should manage the dependencies by yourself, pip install [library name] should be enough for all libraries, check the imports at the begining of the notebooks
