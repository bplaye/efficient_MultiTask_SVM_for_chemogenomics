import pickle
import numpy as np


class experiment_template():
    """
    parameters:
    -----------
    prot_kernel_name : string, name of the protein kernel
    mol_kernel_name : string, name of the molecule kernel
    type_clf : string, type of classifier ('SVM' or 'KernelRidge')
    nb_fold : nb of folds in the cross validation scheme, either 5 or 10
    
    attributes:
    -----------
    list_param : list of real values, list of parameter C (for SVM) or alpha (for KernelRidge)
    nb_sampling : 4, the number of different training sets
    name : string, name of the experiment
    K_prot : array, kernel on proteins
    dico_indice2prot : dict, keys are indices, values are protein IDs
    dico_prot2indice : dict, keys are molecule IDs, values are indices
    K_mol : array, kernel on molecules
    dico_indice2mol : dict, keys are indices, values are molecule IDs
    dico_mol2indice : dict, keys are molecule IDs, values are indices
    """
    def __init__(self, prot_kernel_name, mol_kernel_name, type_clf, nb_fold):
        self.type_clf = type_clf
        self.nb_sampling = 4
        self.nb_fold = nb_fold
        self.name = None
        
        if self.type_clf=="SVM":
            self.list_param = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        elif self.type_clf=="KernelRidge":
            self.list_param = [50000, 5000, 500, 50, 5, 0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005]
        else:
            raise ValueError("wrong value of type_clf")
            
        self.K_prot = pickle.load(open('data/kernels/'+prot_kernel_name+'.data', 'rb'))
        self.dico_prot2indice = pickle.load(open('data/kernels/dico_2indice_'+prot_kernel_name+'.data', 'rb'))
        self.dico_indice2prot = pickle.load(open('data/kernels/dico_indice2_'+prot_kernel_name+'.data', 'rb'))
        
        self.K_mol = pickle.load(open('data/kernels/'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
        self.dico_indice2mol = pickle.load(open('data/kernels/dico_indice2_'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
        self.dico_mol2indice = pickle.load(open('data/kernels/dico_2indice_'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
        
    def make_Ktrain_and_Ktest_ST(self, list_train_samples, list_test_samples, Kernel, dico_2indice):
        """
        computes the kernels associated respectively to the training and testing sets when considering only molecules or proteins
        """
        K_train = np.zeros((len(list_train_samples),len(list_train_samples)))
        K_test = np.zeros((len(list_test_samples),len(list_train_samples)))

        for ind1 in range(len(list_train_samples)):
            if list_train_samples[ind1] in list_test_samples:
                raise ValueError("exit because train is in test")
            ind1_K = dico_2indice[list_train_samples[ind1]]
            for ind2 in range(len(list_train_samples)):
                ind2_K = dico_2indice[list_train_samples[ind2]]
                K_train[ind1, ind2] = Kernel[ind1_K, ind2_K]
                K_train[ind2, ind1] = K_train[ind1, ind2]
            for ind_test in range(len(list_test_samples)):
                ind_test_K = dico_2indice[list_test_samples[ind_test]]
                K_test[ind_test, ind1] = Kernel[ind_test_K, ind1_K]

        return K_train, K_test

    def make_Ktrain_and_Ktest_MT(self, list_train_samples, list_test_samples):
        """
        computes the kernels associated respectively to the training and testing sets when working on the chemogenomic space.
        (i.e. the kronecker over the molecular and protein spaces)
        """
        K_train = np.zeros((len(list_train_samples), len(list_train_samples)))
        K_test = np.zeros((len(list_test_samples),len(list_train_samples)))
        for ind1 in range(len(list_train_samples)):
            if list_train_samples[ind1] in list_test_samples:
                raise ValueError("exit because train is in test")
            ind1_Kprot = self.dico_prot2indice[list_train_samples[ind1][0]]
            ind1_Kmol = self.dico_mol2indice[list_train_samples[ind1][1]]
            for ind2 in range(ind1,len(list_train_samples)):
                ind2_Kprot = self.dico_prot2indice[list_train_samples[ind2][0]]
                ind2_Kmol = self.dico_mol2indice[list_train_samples[ind2][1]]
                K_train[ind1,ind2] = self.K_prot[ind1_Kprot, ind2_Kprot] * self.K_mol[ind1_Kmol, ind2_Kmol]
                K_train[ind2,ind1] = K_train[ind1,ind2]
            for ind_t in range(len(list_test_samples)):
                ind_t_Kprot = self.dico_prot2indice[list_test_samples[ind_t][0]]
                ind_t_Kmol = self.dico_mol2indice[list_test_samples[ind_t][1]]
                K_test[ind_t, ind1] = self.K_prot[ind1_Kprot, ind_t_Kprot] * self.K_mol[ind1_Kmol,ind_t_Kmol]

        return K_train, K_test
