import pickle
from src.kernels_list import *
from src.experiment_template import *
import sys
import csv
import collections
import copy

from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn import metrics
    
class MT_KernelAnalysis_experiment(experiment_template):
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
        experiment_template.__init__(self,prot_kernel_name, mol_kernel_name, type_clf, nb_fold)
        
    def load_CV_indexes(self, ind_sampling):
        self.samples_tr = pickle.load(open('data/pickles/MT/'+str(self.nb_fold)+'fold_samples_tr_S1_'+str(ind_sampling)+'.data', 'rb'))
        self.labels_tr = pickle.load(open('data/pickles/MT/'+str(self.nb_fold)+'fold_labels_tr_S1_'+str(ind_sampling)+'.data', 'rb'))
        self.samples_te = pickle.load(open('data/pickles/MT/'+str(self.nb_fold)+'fold_samples_te_S1_'+str(ind_sampling)+'.data', 'rb'))
        self.labels_te = pickle.load(open('data/pickles/MT/'+str(self.nb_fold)+'fold_labels_te_S1_'+str(ind_sampling)+'.data', 'rb'))
        
    def run(self):
        """
        outputs are pickles: list of list of real values, for each parameter of the clf, list of scores on each outer-folds or inner-folds
        """
        list_mean_auc = [[] for param in range(len(self.list_param))]
        list_mean_aupr = [[] for param in range(len(self.list_param))]
        list_mean_auc = [[] for param in range(len(self.list_param))]
        list_mean_aupr = [[] for param in range(len(self.list_param))]

        for ind_sampling in range(self.nb_sampling):
            print("ind_sampling=", ind_sampling)
            sys.stdout.flush()

            mean_fold_auc = [[] for param in range(len(self.list_param))]
            mean_fold_aupr = [[] for param in range(len(self.list_param))]
            mean_fold_auc_nested = [[] for param in range(len(self.list_param))]
            mean_fold_aupr_nested = [[] for param in range(len(self.list_param))]

            self.load_CV_indexes(ind_sampling)

            for ind_fold in range(self.nb_fold):
                    print("ind_fold=", ind_fold)
                    sys.stdout.flush()

                    ##start computation in outside CV
                    K_train, K_test = self.make_Ktrain_and_Ktest_MT(self.samples_tr[ind_fold], self.samples_te[ind_fold])

                    for param in range(len(self.list_param)):
                        if self.type_clf=="SVM":
                            clf = svm.SVC(kernel='precomputed', C=self.list_param[param])
                        else:
                            raise ValueError('invalid value of type_clf')
                        clf.fit(K_train, self.labels_tr[ind_fold])
                        Y_test_score = clf.decision_function(K_test).tolist()

                        mean_fold_auc[param].append(metrics.roc_auc_score(self.labels_te[ind_fold], Y_test_score))
                        mean_fold_aupr[param].append(metrics.average_precision_score(self.labels_te[ind_fold],  Y_test_score))

                        del clf
                        del Y_test_score

                    del K_train
                    del K_test



                    ## start computation in inner CV
                    inner_samples_tr = []
                    inner_labels_tr = []
                    inner_samples_te = []
                    inner_labels_te = []
                    for ind_t in range(self.nb_fold):
                        if ind_t!=ind_fold:
                            list_pos_local = []
                            list_neg_local = []
                            for ind_temp in range(len(self.samples_te[ind_t])):
                                if self.labels_te[ind_t][ind_temp] == 1:
                                    list_pos_local.append(self.samples_te[ind_t][ind_temp])
                                else:
                                    list_neg_local.append(self.samples_te[ind_t][ind_temp])
                            inner_samples_te.append(list_pos_local+list_neg_local)
                            inner_labels_te.append([1 for _ in range(len(list_pos_local))] + [-1 for _ in range(len(list_neg_local))])
                    for ind_temp in range(self.nb_fold-1):
                            list_temp_sample = []
                            list_temp_label = []
                            for ind_temp_bis in range(self.nb_fold-1):
                                if ind_temp!=ind_temp_bis:
                                    list_temp_sample += inner_samples_te[ind_temp_bis]
                                    list_temp_label += inner_labels_te[ind_temp_bis]
                            inner_samples_tr.append(list_temp_sample)
                            inner_labels_tr.append(list_temp_label)


                    for ind_fold_nested in range(self.nb_fold-1):
                        print("ind_fold_nested=", ind_fold_nested)
                        sys.stdout.flush()
                        K_train, K_test = self.make_Ktrain_and_Ktest_MT(inner_samples_tr[ind_fold_nested], inner_samples_te[ind_fold_nested])

                        for param in range(len(self.list_param)):
                            if self.type_clf=="SVM":
                                clf = svm.SVC(kernel='precomputed', C=self.list_param[param])
                            elif self.type_clf=="RF":
                                raise ValueError('invalid value of type_clf')
                            clf.fit(K_train, inner_labels_tr[ind_fold_nested])
                            Y_test_score = clf.decision_function(K_test).tolist()

                            mean_fold_auc_nested[param].append(metrics.roc_auc_score(inner_labels_te[ind_fold_nested], Y_test_score))
                            mean_fold_aupr_nested[param].append(metrics.average_precision_score(inner_labels_te[ind_fold_nested], Y_test_score))

                            del clf
                            del Y_test_score

                        del K_train
                        del K_test

            for param in range(len(self.list_param)):#enregistre les résultats pour les différents list_partial_neg
                list_mean_auc[param].append(mean_fold_auc[param])
                list_mean_aupr[param].append(mean_fold_aupr[param])
                list_mean_auc_nested[param].append(mean_fold_auc_nested[param])
                list_mean_aupr_nested[param].append(mean_fold_aupr_nested[param])

        pickle.dump(list_mean_auc_nested, open("saved_results/MT_KernelAnalysis/ChoiceKernel_S1_SVM_auc-nested_"+sys.argv[1]+"_"+sys.argv[2]+".data.data", 'wb'))
        pickle.dump(list_mean_aupr_nested, open("saved_results/MT_KernelAnalysis/ChoiceKernel_S1_SVM_aupr-nested_"+sys.argv[1]+"_"+sys.argv[2]+".data", 'wb'))
        pickle.dump(list_mean_auc, open("saved_results/MT_KernelAnalysis/ChoiceKernel_S1_SVM_auc_"+sys.argv[1]+"_"+sys.argv[2]+".data", 'wb'))
        pickle.dump(list_mean_aupr, open("saved_results/MT_KernelAnalysis/ChoiceKernel_S1_SVM_aupr_"+sys.argv[1]+"_"+sys.argv[2]+".data", 'wb'))

      
if __name__ == '__main__':
    list_mol_kernel_name = make_mol_kernel_list()
    list_prot_kernel_name = make_prot_kernel_list()
    ind_mol_kernel = int(sys.argv[1])
    ind_prot_kernel = int(sys.argv[2])
    prot_kernel_name = list_prot_kernel_name[ind_prot_kernel]
    mol_kernel_name = list_mol_kernel_name[ind_mol_kernel]
    
    ## best couple of kernels
#    mol_kernel_name = 'Tanimoto_d=8'
#    prot_kernel_name = 'allDrugBankHumanTarget_Profile_CenteredNormalized_k5_threshold7.5'
#    print(list_prot_kernel_name.index(prot_kernel_name))
#    print(list_mol_kernel_name.index(mol_kernel_name))
#    exit(1)
    
    experiment = MT_KernelAnalysis_experiment(prot_kernel_name, mol_kernel_name, 'SVM', 5)
    experiment.run()
    
