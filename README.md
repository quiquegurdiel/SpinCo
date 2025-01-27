# SpinCo
Spindle detection application for Cognition database

### Required steps for setup:
1- clone repository to [your folder]

2- create a virtual environment with Python 3.10 called SpinCo and, inside the environment, install dependencies running "pip install -r requirements.txt"

3- edit lab.env in the repository folder file like:

    DATAPATH='[the path to COGNITION database in your computer]'

4- add [your folder]/src to the path environment variable of your system manually (in order to use import of the library)

### Demo:

1- Download publicly available DREAMS database from "Stephanie Devuyst. (2005). The DREAMS Databases and Assessment Algorithm [Data set]. Zenodo. Specifically, download "DatabaseSpindles.rar" from https://doi.org/10.5281/zenodo.2650142".

2- Extract the .rar content to [your folder]/demo/DREAMS. 

3- Run the notebooks of SpinCo/demo in order (000_DataSplit, 001_FeatureExtraction, 002_Training, 003_ValidationperModel_Testing_E1_IoU_0.2, 003_ValidationperModel_Testing_E1_IoU_0.3).

For the sake of completeness, we have also included in SpinCo/demo  pickle files (".pkl") with the feature selection results and XGBoost models configuration, as well as an experiments subfolder with the trained XGBoost models, so the final users can check and replicate the results by just running 003_ValidationperModel_Testing_E1_IoU_0.2 and 003_ValidationperModel_Testing_E1_IoU_0.3 Jupyter Notebooks.

### Cite:
If you use this code kindly cite:
"Beyond the ground truth, XGBoost model applied to sleep spindle event detection" E. Gurdiel, J. Gómez-Pilar, F. Vaquerizo-Villar, G. C. Gutiérrez-Tobal, F. del Campo, and R. Hornero. (To Be Published)

### Disclaimer:
This is research code, is distributed with no guaranty. No model, signal or feature vector is shared in this repository. This repository is not designed to be used in a production environment. The intention is to facilitate the actual code used in the research, not a ready-to-use product.

For any doubt regarding funcionality contact quiquegurdiel (at) gmail.com.

You should manage the dependencies by yourself, pip install [library name] should be enough for all libraries, check the imports at the begining of the notebooks.

Using virtual environmets is higly recommended.

### Structure of the repository:
Under src folder there is file named spinco.py that is the library used in the notebooks. The file is separated in different sections to facilitate the search for specific functions, in particular:
- Helpers
- Mathematics
- Filtering
- Band statistic
- Time domain input
- Frequency domain input
- Time-window metrics
- Spindle Detection
- Database management
- Feature & label management
- Experiment & model management
- Metrics

We recomend to run the demo and analise the functions that are called in order to understand the structure of the actual research notebooks, reading straight from the library might be less intuitive and could en up in reading code that is not actually used on the part of interest of the reader.

### Feature extraction:
The notebooks used for feature extraction are right under SpinCo folder and use the prefixes "Extraction_". The feature extraction notebooks include all the steps required for preprocessing.

### Feature selection:
Feature selection notebook is right under SpinCo folder in the file "FeatureSelection_Bootstrap_DREAMS.ipynb". Feature selection results are also stored in pickle files (".pkl")

### Experiments:
Under SpinCo/experiments folder there are several experiments, the results published are computed in the following experiments that have a subfolder associated:
- MASS_Nsel_250: main results of MASS database
- COGNITION_Nsel_250: main results of COGNITION database
- MASS_Nsel_500: Performance comparison of SpinCo for different subsets of features
- MASS_Nsel_750: Performance comparison of SpinCo for different subsets of features
- MASS_Nsel_900: Performance comparison of SpinCo for different subsets of features
- MASS_Nsel_1000: Performance comparison of SpinCo for different subsets of features.
- COGNITION_Nsel_100: Performance comparison of SpinCo for different subsets of features.
- COGNITION_Nsel_500: Performance comparison of SpinCo for different subsets of features.
- COGNITION_Nsel_750: Performance comparison of SpinCo for different subsets of features.
- COGNITION_Nsel_900: Performance comparison of SpinCo for different subsets of features.
- COGNITION_Nsel_1000: Performance comparison of SpinCo for different subsets of features.
- COGNITION_Nsel_x: Performance comparison of SpinCo for different subsets of features.

COGNITION-related subfolders contain jupyter notebooks (".ipynb") for data split (000_DataSplit), XGBoost models training (001_Training), and validation and testing (002_ValidationperModel_Testing_E1_IoU_0.2 and 002_ValidationperModel_Testing_E1_IoU_0.3). COGNITION_Nsel_250 subfolder also contains Jupyter notebooks for per-subject spindles characteristics analysis (003_Correlations_E1_IoU_0.2 and 003_Correlations_E1_IoU_0.3) and validation and testing using classical F1 metric (004_ValidationperModel_Testing_E1_IoU_0.2_oldF1 and 004_ValidationperModel_Testing_E1_IoU_0.3_oldF1). In all the COGNITION-related subfolders, there are also pickle files (".pkl") with the feature selection results and XGBoost models configuration.

MASS-related subfolders contain Jupyter notebooks (".ipynb") for data split (000_DataSplit), XGBoost models training (001_Training), validation and testing (002_ValidationperModel_Testing_E1_IoU_0.2, 002_ValidationperModel_Testing_E1_IoU_0.3, 002_ValidationperModel_Testing_E2_IoU_0.2, and 002_ValidationperModel_Testing_E2_IoU_0.3).  MASS_Nsel_250 subfolder also contains Jupyter notebooks for per-subject spindles characteristics analysis (003_Correlations_E1_IoU_0.2, 003_Correlations_E1_IoU_0.3, 003_Correlations_E2_IoU_0.2, and 003_Correlations_E2_IoU_0.3), an inter-expert variability and inter-expert generalization test (004_Interpredictability), and inter-expert per-subject spindles characteristics analysis (005_InterExpertCorrelations_E1_IoU_0.2, 005_InterExpertCorrelations_E1_IoU_0.3, 005_InterExpertCorrelations_E2_IoU_0.2, and 005_InterExpertCorrelations_E2_IoU_0.3), and validation and testing using classical F1 metric (006_ValidationperModel_Testing_E1_IoU_0.2_oldF1, and 006_ValidationperModel_Testing_E1_IoU_0.3_oldF1, 006_ValidationperModel_Testing_E2_IoU_0.2_oldF1, and 006_ValidationperModel_Testing_E2_IoU_0.3_oldF1). In all the MASS-related subfolders, there are also pickle files (".pkl") with the feature selection results and XGBoost models configuration.

Importantly, these experiments are based on the EEG features previously extracted (see the Feature extraction section of the README) and selected (see the Feature selection section of the README). 

### Contact:
Feel free to reach out to us with any questions regarding the code repository or databases. You can contact Enrique Gurdiel at enrique.gurdiel@gib.tel.uva.es and/or Fernando Vaquerizo-Villar at fernando.vaquerizo@uva.es.

