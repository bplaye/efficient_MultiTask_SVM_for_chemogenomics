from src.experiment_template import *
import sys
import pickle    
import csv
import collections
import copy
import math

from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn import metrics

class MT_experiment(experiment_template):
    """
    compute n-foldCV performances of a specific "clf, K_prot, K_mol, fold_setting, CV_type"
    attributes:
    -----------
    list_param : list of real values, list of parameter C (for SVM) or alpha (for KernelRidge)
    nb_sampling : 4, the number of different training sets
    name : string, name of the experiment
    K_prot : array, kernel on proteins, 'Profile_CenteredNormalized_k5_threshold7.5'
    dico_indice2prot : dict, keys are indices, values are protein IDs
    dico_prot2indice : dict, keys are molecule IDs, values are indices
    K_mol : array, kernel on molecules, 'Tanimoto_d=8'
    dico_indice2mol : dict, keys are indices, values are molecule IDs
    dico_mol2indice : dict, keys are molecule IDs, values are indices
    
    parameters:
    -----------
    type_clf : string, type of classifier ('SVM' or 'KernelRidge')
    nb_fold : nb of folds in the cross validation scheme, either 5 or 10
    fold_setting: string, in ['S1','S2','S3','S4']
    CV_type: string, either '' or 'ClusterCV_'
    """
    def __init__(self, type_clf, nb_fold, fold_setting, CV_type):
        mol_kernel_name = 'Tanimoto_d=8'
        prot_kernel_name = 'allDrugBankHumanTarget_Profile_CenteredNormalized_k5_threshold7.5'
        experiment_template.__init__(self, prot_kernel_name, mol_kernel_name, type_clf, nb_fold)
        self.fold_setting = fold_setting
        self.CV_type = CV_type
     
    def load_CV_indexes(self, ind_sampling):
        self.samples_tr = pickle.load(open('data/pickles/MT/'+self.CV_type+str(self.nb_fold)+'fold_samples_tr_'+self.fold_setting+'_'+str(ind_sampling)+'.data', 'rb'))
        self.labels_tr = pickle.load(open('data/pickles/MT/'+self.CV_type+str(self.nb_fold)+'fold_labels_tr_'+self.fold_setting+'_'+str(ind_sampling)+'.data', 'rb'))
        self.samples_te = pickle.load(open('data/pickles/MT/'+self.CV_type+str(self.nb_fold)+'fold_samples_te_'+self.fold_setting+'_'+str(ind_sampling)+'.data', 'rb'))
        self.labels_te = pickle.load(open('data/pickles/MT/'+self.CV_type+str(self.nb_fold)+'fold_labels_te_'+self.fold_setting+'_'+str(ind_sampling)+'.data', 'rb'))
        
    def make_Ktrain_and_Ktest_MT_with_settings(self,list_samples_tr, list_samples_te):
        list_mol_in_test = []
        list_prot_in_test = []
        for cou in list_samples_te:
            if cou[0] not in list_prot_in_test:
                list_prot_in_test.append(cou[0])
            if cou[1] not in list_mol_in_test:
                list_mol_in_test.append(cou[1])

        K_train = np.zeros((len(list_samples_tr), len(list_samples_tr)))
        K_test = np.zeros((len(list_samples_te),len(list_samples_tr)))
        for ind1 in range(len(list_samples_tr)):
            if self.fold_setting=="S1":
                if list_samples_tr[ind1] in list_samples_te:
                    raise ValueError("exit because train is in test")
            elif self.fold_setting=="S2":
                if list_samples_tr[ind1][1] in list_mol_in_test:
                    raise ValueError("exit because train is in test")
            elif self.fold_setting=="S3":
                if list_samples_tr[ind1][0] in list_prot_in_test:
                    raise ValueError("exit because train is in test")
            elif self.fold_setting=="S4":
                if list_samples_tr[ind1][0] in list_prot_in_test or list_samples_tr[ind1][1] in list_mol_in_test:
                    raise ValueError("exit because train is in test")
            ind1_Kprot = self.dico_prot2indice[list_samples_tr[ind1][0]]
            ind1_Kmol = self.dico_mol2indice[list_samples_tr[ind1][1]]
            for ind2 in range(ind1,len(list_samples_tr)):
                ind2_Kprot = self.dico_prot2indice[list_samples_tr[ind2][0]]
                ind2_Kmol = self.dico_mol2indice[list_samples_tr[ind2][1]]
                K_train[ind1,ind2] = self.K_prot[ind1_Kprot, ind2_Kprot]*self.K_mol[ind1_Kmol, ind2_Kmol]
                K_train[ind2,ind1] = K_train[ind1,ind2]
            for ind_t in range(len(list_samples_te)):
                ind_t_Kprot = self.dico_prot2indice[list_samples_te[ind_t][0]]
                ind_t_Kmol = self.dico_mol2indice[list_samples_te[ind_t][1]]
                K_test[ind_t, ind1] = self.K_prot[ind1_Kprot, ind_t_Kprot]*self.K_mol[ind1_Kmol,ind_t_Kmol]

        return K_train, K_test

    def run(self, ind_sampling, ind_fold):
        if self.fold_setting=="S4":
            nb_fold = self.nb_fold * self.nb_fold
        self.load_CV_indexes(ind_sampling)
        
        if self.CV_type == 'ClusterCV_':
                ajout = self.CV_type
        else:
            ajout = 'CV_'
        
        K_train, K_test = self.make_Ktrain_and_Ktest_MT_with_settings(self.samples_tr[ind_fold], self.samples_te[ind_fold])
        pred_score = []
        for param in range(len(self.list_param)):
            if self.type_clf=="SVM":
                clf = svm.SVC(kernel='precomputed', C=self.list_param[param])
                clf.fit(K_train, self.labels_tr[ind_fold])
                Y_test_score = clf.decision_function(K_test).tolist()
            elif self.type_clf=="KernelRidge":
                clf = KernelRidge(alpha=self.list_param[param], kernel='precomputed')
                clf.fit(K_train, inner_labels_tr[ind_fold])
                Y_test_score = clf.predict(K_test).tolist()
            else:
                raise ValueError('invalid value of type_clf')
            pred_score.append(Y_test_score)
            del clf
            del Y_test_score
        pickle.dump(pred_score, open('saved_results/MT/MT_'+str(self.nb_fold)+'fold'+ajout+self.fold_setting+"_"+self.type_clf+"_"+str(ind_fold)+"_"+str(ind_sampling)+".data", 'wb'))
        del K_train
        del K_test
        
    
########################################################
########################################################
########################################################

class MT_NestedCV_experiment(MT_experiment):
    """
    compute n-foldNestedCV performances of a specific "clf, K_prot, K_mol, fold_setting, CV_type"
    """
    def run(self, ind_sampling, ind_fold, type_calcul, ind_fold_nested):
        if self.fold_setting=="S4":
            nb_fold = self.nb_fold * self.nb_fold
        self.load_CV_indexes(ind_sampling)
        
        if self.CV_type == 'ClusterCV_':
            ajout = self.CV_type
        else:
            ajout = 'CV_'
                
        if type_calcul=="Outer":
            K_train, K_test = self.make_Ktrain_and_Ktest_MT(self.samples_tr[ind_fold], self.samples_te[ind_fold])
            pred_score = []
            for param in range(len(self.list_param)):
                if self.type_clf=="SVM":
                    clf = svm.SVC(kernel='precomputed', C=self.list_param[param])
                    clf.fit(K_train, self.labels_tr[ind_fold])
                    Y_test_score = clf.decision_function(K_test).tolist()
                elif self.type_clf=="KernelRidge":
                    clf = KernelRidge(alpha=self.list_param[param], kernel='precomputed')
                    clf.fit(K_train, inner_labels_tr[ind_fold])
                    Y_test_score = clf.predict(K_test).tolist()
                else:
                    raise ValueError('invalid value of type_clf')
                pred_score.append(Y_test_score)
                del clf
                del Y_test_score
            pickle.dump(pred_score, open('saved_results/MT/MT_Nested'+str(self.nb_fold)+'fold'+ajout+self.fold_setting+"_"+self.type_clf+'_Nested_'+str(ind_fold)+'_'+str(ind_sampling)+".data", 'wb'))
            del K_train
            del K_test

        elif type_calcul=="Inner":
            inner_samples_tr = []
            inner_labels_tr = []
            inner_samples_te = []
            inner_labels_te = []
            for ind_t in range(self.nb_fold):
                if ind_t!=ind_fold_nested:
                    inner_samples_te.append(self.samples_te[ind_t])
                    inner_labels_te.append(self.labels_te[ind_t])
            for ind_temp in range(self.nb_fold-1):
                list_temp_sample = []
                list_temp_label = []
                for ind_temp_bis in range(self.nb_fold-1):
                    if ind_temp!=ind_temp_bis:
                        list_temp_sample += inner_samples_te[ind_temp_bis]
                        list_temp_label += inner_labels_te[ind_temp_bis]
                inner_samples_tr.append(list_temp_sample)
                inner_labels_tr.append(list_temp_label)

            K_train, K_test = self.make_Ktrain_and_Ktest_MT(inner_samples_tr[ind_fold_nested], inner_samples_te[ind_fold_nested])
            pred_score = []
            for param in range(len(self.list_param)):
                if self.type_clf=="SVM":
                    clf = svm.SVC(kernel='precomputed', C=self.list_param[param])
                    clf.fit(K_train, inner_labels_tr[ind_fold_nested])
                    Y_test_score = clf.decision_function(K_test).tolist()
                elif self.type_clf=="KernelRidge":
                    clf = KernelRidge(alpha=self.list_param[param], kernel='precomputed')
                    clf.fit(K_train, inner_labels_tr[ind_fold_nested])
                    Y_test_score = clf.predict(K_test).tolist()
                else:
                    raise ValueError('invalid value of type_clf')
                pred_score.append(Y_test_score)
                del clf
                del Y_test_score
            pickle.dump(pred_score, open('saved_results/MT/MT_Nested'+str(self.nb_fold)+'fold'+ajout+self.fold_setting+"_"+self.type_clf+'_In_'+str(ind_fold)+'_'+str(ind_fold_nested)+'_'+str(ind_sampling)+".data", 'wb'))
            del K_train
            del K_test
            
########################################################
########################################################
########################################################

class MT_LOO_experiment(MT_experiment):
    """
    compute LOOCV performances of a specific "clf, K_prot, K_mol, fold_setting, CV_type"
    """
    def compute_K_couple(self, list_samples):
        K_couple = np.zeros((len(list_samples), len(list_samples)))
        for ind1 in range(len(list_samples)):
            ind1_Kprot = self.dico_prot2indice[list_samples[ind1][0]]
            ind1_Kmol = self.dico_mol2indice[list_samples[ind1][1]]
            for ind2 in range(ind1,len(list_samples)):
                ind2_Kprot = self.dico_prot2indice[list_samples[ind2][0]]
                ind2_Kmol = self.dico_mol2indice[list_samples[ind2][1]]
                K_couple[ind1,ind2] = self. K_prot[ind1_Kprot, ind2_Kprot] * self.K_mol[ind1_Kmol, ind2_Kmol]
                K_couple[ind2,ind1] = K_couple[ind1,ind2]
        return K_couple
        
    def make_Ktrain_and_Ktest_MTLOO(self, K_couple, list_couples_labels, ind_couple_test, couple_test, list_couples):
        K_train = K_couple.copy()
        K_test = K_couple[ind_couple_test,:].copy()
        list_train_labels = list_couples_labels.copy()

        if self.fold_setting=="S1":
            del list_train_labels[ind_couple_test]
            K_train = np.delete(K_train, ind_couple_test, axis=0)
            K_train = np.delete(K_train, ind_couple_test, axis=1)
            K_test = np.delete(K_test, ind_couple_test)

        elif self.fold_setting=="S2":
            deleted_couples_ind = []
            for ind_couple in range(len(list_couples)):
                if list_couples[ind_couple][1]==couple_test[1]:
                    deleted_couples_ind.append(ind_couple)
            deleted_couples_ind = np.sort(deleted_couples_ind)
            #print(deleted_couples_ind)
            #print(len(deleted_couples_ind))
            for ind_of_ind in range(len(deleted_couples_ind)):
                deleted_ind = deleted_couples_ind[len(deleted_couples_ind)-1-ind_of_ind]
                del list_train_labels[deleted_ind]
                K_train = np.delete(K_train, deleted_ind, axis=0)
                K_train = np.delete(K_train, deleted_ind, axis=1)
                K_test = np.delete(K_test, deleted_ind)

        elif self.fold_setting=="S3":
            deleted_couples_ind = []
            for ind_couple in range(len(list_couples)):
                if list_couples[ind_couple][0]==couple_test[0]:
                    deleted_couples_ind.append(ind_couple)
            #print(len(deleted_couples_ind))
            deleted_couples_ind = np.sort(deleted_couples_ind)
            for ind_of_ind in range(len(deleted_couples_ind)):
                deleted_ind = deleted_couples_ind[len(deleted_couples_ind)-1-ind_of_ind]
                del list_train_labels[deleted_ind]
                K_train = np.delete(K_train, deleted_ind, axis=0)
                K_train = np.delete(K_train, deleted_ind, axis=1)
                K_test = np.delete(K_test, deleted_ind)

        elif self.fold_setting=="S4":
            #print(couple_test)
            #print(list_couples_tot[ind_couple_test])
            deleted_couples_ind = []
            for ind_couple in range(len(list_couples)):
                if list_couples[ind_couple][1]==couple_test[1]:
                    deleted_couples_ind.append(ind_couple)
                elif list_couples[ind_couple][0]==couple_test[0]:
                    deleted_couples_ind.append(ind_couple)
            #print(len(deleted_couples_ind))
            deleted_couples_ind = np.sort(deleted_couples_ind)
            for ind_of_ind in range(len(deleted_couples_ind)):
                deleted_ind = deleted_couples_ind[len(deleted_couples_ind)-1-ind_of_ind]
                #print(list_couples[deleted_ind])
                del list_train_labels[deleted_ind]
                K_train = np.delete(K_train, deleted_ind, axis=0)
                K_train = np.delete(K_train, deleted_ind, axis=1)
                K_test = np.delete(K_test, deleted_ind)

        return K_train, K_test, list_train_labels
    
    def load_list_couples(self,ind_sampling):
        list_pos_couples = pickle.load(open('data/pickles/MT/list_pos_couples.data', 'rb'))
        list_partial_neg_couples = pickle.load(open('data/pickles/MT/list_partial_neg_couples.data', 'rb'))
        return list_pos_couples, list_partial_neg_couples[ind_sampling]
    
    def run(self, ind_sampling, ind_partial):
        if self.type_clf=="KernelRidge":
            value_of_neg_class = 0
        else:
            value_of_neg_class = -1
        list_pos_couples, list_neg_couples = self.load_list_couples(ind_sampling)
        
        list_couples_tot = list_pos_couples + list_neg_couples
        list_couples_labels = [1 for _ in range(len(list_pos_couples))] + [value_of_neg_class for _ in range(len(list_neg_couples))]

        nb_couples = math.floor(float(len(list_couples_tot))/float(100))
        if ind_partial!=99:
            list_couples_partial = list_couples_tot[ind_partial*nb_couples:(1+ind_partial)*nb_couples].copy()
        else:
            list_couples_partial = list_couples_tot[ind_partial*nb_couples:].copy()
        print("nb_couples = ", len(list_couples_partial))

        dico_couple2ind = {}
        for ind_couple in range(len(list_couples_tot)):
            dico_couple2ind[list_couples_tot[ind_couple]] = ind_couple
        K_couple = self.compute_K_couple(list_couples_tot)

        list_pred_score = []
        for param in range(len(self.list_param)):
            list_pred_score.append([])

        for ind_couples in range(len(list_couples_partial)):
            print("ind_couples=", ind_couples)
            sys.stdout.flush()

            couple_test = list_couples_partial[ind_couples]
            K_train, K_test, list_train_labels = self.make_Ktrain_and_Ktest_MTLOO(K_couple, list_couples_labels, dico_couple2ind[couple_test], couple_test, list_couples_tot)

            for param in range(len(self.list_param)):
                if self.type_clf=="SVM":
                    clf = svm.SVC(kernel='precomputed', C=self.list_param[param])
                    clf.fit(K_train, list_train_labels)
                    list_pred_score[param]+=clf.decision_function(K_test).tolist()
            del clf
            del K_train
            del K_test
            del list_train_labels

        pickle.dump(list_pred_score, open('saved_results/MT/MultiLOOCV_'+self.fold_setting+'_'+slef.type_clf+'_'+str(ind_partial)+'_'+str(ind_sampling)+'.data', 'wb'))
        
        
if __name__ == '__main__':
    type_clf = sys.argv[2]
    fold_setting = sys.argv[3]
    ind_sampling = int(sys.argv[-1])

    if 'ClusterCV' in sys.argv[1]:
        if 'Nested5foldCV' in sys.argv[1]:
            type_calcul = sys.argv[4]
            ind_fold_nested = int(sys.argv[5])
            ind_fold = int(sys.argv[6])
            experiment = MT_NestedCV_experiment(type_clf, 5, fold_setting, 'ClusterCV_')
            experiment.run(ind_sampling, ind_fold, type_calcul, ind_fold_nested)
        elif '5foldCV' in sys.argv[1]:
            ind_fold = int(sys.argv[4])
            experiment = MT_experiment(type_clf, 5, fold_setting, 'ClusterCV_')
            experiment.run(ind_sampling, ind_fold)
        elif '10foldCV' in sys.argv[1]:
            ind_fold = int(sys.argv[4])
            experiment = MT_experiment(type_clf, 10, fold_setting, 'ClusterCV_')
            experiment.run(ind_sampling, ind_fold)
    elif 'LOOCV' in sys.argv[1]:
        ind_partial = int(sys.argv[4])
        experiment = MT_LOO_experiment(type_clf, None, fold_setting, '')
        experiment.run(ind_sampling, ind_partial)
    elif 'Nested5foldCV' in sys.argv[1]:
        type_calcul = sys.argv[4]
        ind_fold_nested = int(sys.argv[5])
        ind_fold = int(sys.argv[6])
        experiment = MT_NestedCV_experiment(type_clf, 5, fold_setting, '')
        experiment.run(ind_sampling, ind_fold, type_calcul, ind_fold_nested)
    elif '5foldCV' in sys.argv[1]:
        ind_fold = int(sys.argv[4])
        experiment = MT_experiment(type_clf, 5, fold_setting, '')
        experiment.run(ind_sampling, ind_fold)
    elif '10foldCV' in sys.argv[1]:
        ind_fold = int(sys.argv[4])
        experiment = MT_experiment(type_clf, 10, fold_setting, '')
        experiment.run(ind_sampling, ind_fold)
        
        
