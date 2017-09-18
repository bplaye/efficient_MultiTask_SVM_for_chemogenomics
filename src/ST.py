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

def randomize_list(list_to_be_randomized):
    list_to_be_randomized = copy.deepcopy(list_to_be_randomized)
    randomized_list = []
    while len(list_to_be_randomized)>0:
        rand_int = np.random.randint(len(list_to_be_randomized))
        randomized_list.append(list_to_be_randomized[rand_int])
        del list_to_be_randomized[rand_int]
    return randomized_list

class ST_experiment(experiment_template):
    """
    compute LOOCV performances of ST for a specific "clf, K_prot, K_mol" and specific "type_ST, NbNeg"
    parameters:
    ----------
    type_ST : string, in ['S1','S2','S3','S4'], setting
    NbNeg : int, nb of negative instances in train set is "nb of pos instances in train set * NbNeg"
    """
    def __init__(self, type_ST, NbNeg):
        mol_kernel_name = 'Tanimoto_d=8'
        prot_kernel_name = 'allDrugBankHumanTarget_Profile_CenteredNormalized_k5_threshold7.5'
        experiment_template.__init__(self, prot_kernel_name, mol_kernel_name, 'SVM', None)
        self.type_ST = type_ST
        self.NbNeg = NbNeg
        self.name = 'saved_results/ST/ST_LOOCV_'+self.type_ST+'_'+self.type_clf+'_NbNeg='+str(self.NbNeg)
        self.dico_target_of_mol = pickle.load(open('data/pickles/ST/dico_target_of_mol.data', 'rb'))
        self.dico_ligand_of_prot = pickle.load(open('data/pickles/ST/dico_ligand_of_prot.data', 'rb'))
        self.threshold_prot = None
        self.threshold_mol = None
    
    def condition_on_intra_task(self, Kernel, dico_2indice, compared_sample, tested_sample, threshold):
        return True
        
    def make_ST_train_set(self, list_pos, list_neg_samples, list_test_samples, Kernel, dico_2indice, threshold, value_of_neg_class):
        """
        adding intra_task pairs to the train set
        """
        list_train_labels = []
        list_train_samples = []

        for pos_train_sample in list_pos:
            if pos_train_sample not in list_test_samples:
                if self.condition_on_intra_task(Kernel, dico_2indice, pos_train_sample, list_test_samples[0], threshold):
                    list_train_samples.append(pos_train_sample)
                    list_train_labels.append(1)

        if self.NbNeg!="full":
            nb_pos_sample=len(list_train_samples)
            nb_neg_sample=0
            rand_list_of_indices = randomize_list([i for i in range(len(list_neg_samples))])
            for rand_int in rand_list_of_indices:
                if list_neg_samples[rand_int] not in list_pos and list_neg_samples[rand_int] not in list_test_samples and list_neg_samples[rand_int] not in list_train_samples and nb_neg_sample<nb_pos_sample * self.NbNeg:
                    list_train_samples.append(list_neg_samples[rand_int])
                    list_train_labels.append(value_of_neg_class)
                    nb_neg_sample+=1
        else:
            for neg_sample in list_neg_samples:
                if neg_sample not in list_test_samples and neg_sample not in list_pos and neg_sample not in list_train_samples:
                    list_train_samples.append(neg_sample)
                    list_train_labels.append(value_of_neg_class)
        #print(list_train_samples)
        return list_train_samples, list_train_labels
        
        
    def make_prediction(self, couple_test, value_of_neg_class):
        print('ST make_prediction')
        if self.type_ST=="MolView":
            list_train_samples, list_train_labels = self.make_ST_train_set(self.dico_target_of_mol[couple_test[1]][0], self.dico_target_of_mol[couple_test[1]][1], [couple_test[0]], self.K_prot, self.dico_prot2indice, self.threshold_prot, value_of_neg_class)
            K_train, K_test = self.make_Ktrain_and_Ktest_ST(list_train_samples, [couple_test[0]], self.K_prot, self.dico_prot2indice)

            Y_test_score = []
            for param in range(len(self.list_param)):
                if self.type_clf=="SVM":
                    clf = svm.SVC(kernel='precomputed', C=self.list_param[param], class_weight='balanced')
                    clf.fit(K_train, list_train_labels)
                    Y_test_score.append(clf.decision_function(K_test).tolist())
                elif self.type_clf=="KernelRidge":
                    clf = KernelRidge(alpha=self.list_param[param], kernel='precomputed')
                    clf.fit(K_train, list_train_labels)
                    Y_test_score.append(clf.predict(K_test).tolist())

        elif self.type_ST=="Kron":
            list_train_samples = []
            list_train_labels = []
            list_train_samples_mol, list_train_labels_mol = self.make_ST_train_set(self.dico_target_of_mol[couple_test[1]][0], self.dico_target_of_mol[couple_test[1]][1], [couple_test[0]], self.K_prot, self.dico_prot2indice, self.threshold_prot, value_of_neg_class)
            list_train_samples_prot, list_train_labels_prot = self.make_ST_train_set(self.dico_ligand_of_prot[couple_test[0]][0], self.dico_ligand_of_prot[couple_test[0]][1], [couple_test[1]], self.K_mol, self.dico_mol2indice, self.threshold_mol, value_of_neg_class)

            for el in list_train_samples_mol:
                list_train_samples.append((el,couple_test[1]))
            for el in list_train_samples_prot:
                list_train_samples.append((couple_test[0], el))
            list_train_labels = list_train_labels_mol + list_train_labels_prot
            
            list_train_samples, list_train_labels = self.update_with_extra_task_pairs(couple_test, list_train_samples, list_train_labels)

            K_train, K_test = self.make_Ktrain_and_Ktest_MT(list_train_samples, [couple_test])

            Y_test_score = []
            for param in range(len(self.list_param)):
                if self.type_clf=="SVM":
                    clf = svm.SVC(kernel='precomputed', C=self.list_param[param], class_weight='balanced')
                    clf.fit(K_train, list_train_labels)
                    Y_test_score.append(clf.decision_function(K_test).tolist())
                elif self.type_clf=="KernelRidge":
                    clf = KernelRidge(alpha=self.list_param[param], kernel='precomputed')
                    clf.fit(K_train, list_train_labels)
                    Y_test_score.append(clf.predict(K_test).tolist())

        return Y_test_score
    
    def update_with_extra_task_pairs(self, couple_test, list_train_samples, list_train_labels):
        print('ST extra task')
        sys.stdout.flush()
        return list_train_samples, list_train_labels
    
    def load_list_couples(self, ind_sampling, value_of_neg_class):
        list_couples_tot = pickle.load(open('data/pickles/ST/list_couples_tot_'+str(ind_sampling)+'.data', 'rb'))
        list_couples_labels = pickle.load(open('data/pickles/ST/list_labels_tot_'+str(ind_sampling)+'.data', 'rb'))
        return list_couples_tot, list_couples_labels
    
    def run(self, ind_partial, ind_sampling):
        if self.type_clf=="KernelRidge":
            value_of_neg_class = 0
        else:
            value_of_neg_class = 0

        list_couples_tot, list_couples_labels = self.load_list_couples(ind_sampling, value_of_neg_class)
        nb_couples = math.floor(float(len(list_couples_tot))/float(50))
        if int(ind_partial)!=49:
            list_couples_partial = list_couples_tot[ind_partial*nb_couples:(1+ind_partial)*nb_couples].copy()
        else:
            list_couples_partial = list_couples_tot[ind_partial*nb_couples:].copy()
        print("nb_couples = ", len(list_couples_partial))

        list_pred_score = []
        for param in range(len(self.list_param)):
            list_pred_score.append([])

        for ind_couples in range(len(list_couples_partial)):

            couple_test = list_couples_partial[ind_couples]
            Y_test_score =self.make_prediction(couple_test, value_of_neg_class)

            for param in range(len(self.list_param)):
                list_pred_score[param]+=Y_test_score[param]
        pickle.dump(list_pred_score, open(self.name+'_'+str(ind_partial)+'_'+str(ind_sampling)+'.data', 'wb'))




class ST_minus_experiment(ST_experiment):
    """
    negative intra-task pairs are added conditionnaly
    parameters:
    -----------
    type_ST : string, in ['S1','S2','S3','S4'], setting
    NbNeg : int, nb of negative instances in train set is "nb of pos instances in train set * NbNeg"
    centile : int between 0 and 100, centile of the distribution of mol and prot similarity
        Gives a maximum threshold of similarity between the tested sample and samples in the train set
    nb_partial : nb of subdivision of the computation
    """
    def __init__(self, type_ST, NbNeg, centile, nb_partial=50):
        ST_experiment.__init__(self,type_ST, NbNeg)
        self.centile = centile
        self.name_pos = "saved_results/ST/STminus_LOOCV_pos_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+"_Sd:"+str(self.centile)
        self.name_neg = "saved_results/ST/STminus_LOOCV_neg_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+"_Sd:"+str(self.centile)
        self.nb_partial = nb_partial
        
    def load_list_couples(self, ind_sampling, value_of_neg_class):
        list_couples_tot_pos = pickle.load(open('data/pickles/STminus/list_couples_tot_pos_'+str(ind_sampling)+'_Sd:'+str(self.centile)+'.data', 'rb'))
        list_couples_tot_neg = pickle.load(open('data/pickles/STminus/list_couples_tot_neg_'+str(ind_sampling)+'_Sd:'+str(self.centile)+'.data', 'rb'))
        threshold_per_centile_mol = pickle.load(open('data/pickles/STminus/threshold_per_centile_mol.data', 'rb'))
        threshold_per_centile_prot = pickle.load(open('data/pickles/STminus/threshold_per_centile_prot.data', 'rb'))
        self.threshold_mol = threshold_per_centile_mol[self.centile]
        self.threshold_prot = threshold_per_centile_prot[self.centile]
        
        list_couples_tot = list_couples_tot_pos + list_couples_tot_neg
        list_couples_labels = [1 for i in range(len(list_couples_tot_pos))] + [value_of_neg_class for i in range(len(list_couples_tot_neg))]
        return list_couples_tot_pos, list_couples_tot_neg
        
    def condition_on_intra_task(self, Kernel, dico_2indice, compared_sample, tested_sample, threshold):    
#        print('intra task conditionned')
#        sys.stdout.flush()
        return Kernel[dico_2indice[compared_sample], dico_2indice[tested_sample]]<=threshold
   
    def run(self, ind_partial, ind_sampling):
        if self.type_clf=="KernelRidge":
            value_of_neg_class = 0
        else:
            value_of_neg_class = 0

        list_couples_tot_pos, list_couples_tot_neg = self.load_list_couples(ind_sampling, value_of_neg_class)
        nb_couples = math.floor(float(len(list_couples_tot_pos))/float(self.nb_partial))
        print(nb_couples)
        if int(ind_partial)!=self.nb_partial-1:
            list_couples_pos_partial = list_couples_tot_pos[ind_partial*nb_couples:(1+ind_partial)*nb_couples].copy()
            list_couples_neg_partial = list_couples_tot_neg[ind_partial*nb_couples:(1+ind_partial)*nb_couples].copy()
        else:
            list_couples_pos_partial = list_couples_tot_pos[ind_partial*nb_couples:].copy()
            list_couples_neg_partial = list_couples_tot_neg[ind_partial*nb_couples:].copy()

        list_pred_score = []
        for param in range(len(self.list_param)):
            list_pred_score.append([])
        for ind_couples in range(len(list_couples_pos_partial)):
            couple_test = list_couples_pos_partial[ind_couples]
            print(couple_test)
            Y_test_score =self.make_prediction(couple_test, value_of_neg_class)
            for param in range(len(self.list_param)):
                list_pred_score[param]+=Y_test_score[param]
        pickle.dump(list_pred_score, open(self.name_pos+"_"+str(ind_partial)+"_"+str(ind_sampling)+'.data', 'wb'))

        list_pred_score = []
        for param in range(len(self.list_param)):
            list_pred_score.append([])
        for ind_couples in range(len(list_couples_neg_partial)):
            couple_test = list_couples_neg_partial[ind_couples]
            Y_test_score =self.make_prediction(couple_test, value_of_neg_class)
            for param in range(len(self.list_param)):
                list_pred_score[param]+=Y_test_score[param]
        pickle.dump(list_pred_score, open(self.name_neg+"_"+str(ind_partial)+"_"+str(ind_sampling)+'.data', 'wb'))

if __name__ == '__main__':     
    
    if 'Kron' in sys.argv[1]:
        type_ST = 'Kron'
    elif 'MolView' in sys.argv[1]:
        type_ST = 'MolView'
    if sys.argv[2]=='full':
        NbNeg = 'full'
    else:    
        NbNeg = int(sys.argv[2])
    ind_partial = int(sys.argv[3])
    ind_sampling = int(sys.argv[-1])    
    
    if 'minus' in sys.argv[1]:
        centile = int(sys.argv[4])
        experiment = ST_minus_experiment(type_ST, NbNeg, centile)
        experiment.run(ind_partial, ind_sampling)            
    else:
        experiment = ST_experiment(type_ST, NbNeg)
        experiment.run(ind_partial, ind_sampling)
        
        
