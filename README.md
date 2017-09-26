# Efficient Multi-Task chemogenomics for drug specificity prediction
------------------------------------------------

This is the code to reproduce all findings from the following paper:

Benoit, Playe and Chloé-Agathe, Azencott and Véronique, Stoven. "Efficient Multi-Task chemogenomics for drug specificity prediction." 
link to paper
"
Adverse drug reactions (also called side effects) often interrupt the long and costly process of drug development. Side effects occur when drugs bind to proteins other than their intended target. As experimentally testing drug specificity against the entire proteome is out of reach, we investigate in this paper the application of chemogenomics approaches.

We formulate the study of drug specificity as a proteome-wide drug-protein interaction prediction problem, and build appropriate data sets on which we evaluate multi-task support vector machines.

Our observations lead us to propose NNMT, an efficient Multi-Task SVM for chemogenomics that is trained on a limited number of data points. Finally, we demonstrate the suitability of NNMT to study the specificity of drug-like molecules in real-life situations by suggesting secondary targets for 36 recently withdrawn drugs. In 9 cases, we identified secondary targets responsible for the withdrawal of the drug.
"

The rest of required files can be found at http://members.cbio.mines-paristech.fr/~bplaye/efficient_MT_chemo.tar.gz

************************
Required Python Packages
************************

numpy (>= 1.11.1)

scipy (>= 0.18.1)

matplotlib (>= 1.3.1)

sklearn (>= 0.18.1)

***********
Run scripts
***********

Running the python notebooks to reproduce all figures and table.
Table 2 has been generated with notebook kernel_analysis.ipynb. Figures 1 and 2 have been generated with notebook MT_analysis.ipynb. Figures 3 to 8 have been generated with notebook ST&NMT_analysis.ipynb. Figures 9 to 11 have been generated with notebook family_analysis.ipynb. 
Classes documentation details the use of each kind of performed experiments.

**************
Archive Folder
**************

> src: Contains all experiments classes run for the paper.

> study_on_withdraw_molecules: contains list of considered withdrawn molecules and the list of predicted target for each drugs in the drugbank database.
