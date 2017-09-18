from src.ST import *
import sys


class NNMT_experiment(ST_experiment):
    """
    compute LOOCV performances of NNMT for a specific "clf, K_prot, K_mol" and specific "type_ST, NbNeg, PosNei, NegNei"
    parameters:
    ----------
    type_ST : string, in ['S1','S2','S3','S4'], setting
    NbNeg : int, nb of negative instances in train set is "nb of pos instances in train set * NbNeg"
    PosNei : number of positive extra-task instances in the train set
    NegNei : number of negative extra-task instances in the train set is "NegNei*PosNei"
    """
    def __init__(self, type_ST, NbNeg, PosNei, NegNei):
        ST_experiment.__init__(self, type_ST, NbNeg)
        self.name = 'saved_results/ST/MTNeerestNei_'+self.type_ST+'_S1_'+self.type_clf+'_NbNeg='+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)
        self.PosNei = PosNei
        self.NegNei = NegNei
        self.dico_neerest_nei_per_mol_ID = pickle.load(open('data/pickles/ST/dico_neerest_nei_per_mol_ID.data','rb'))
        self.dico_neerest_nei_per_mol_value = pickle.load(open('data/pickles/ST/dico_neerest_nei_per_mol_value.data','rb'))
        self.dico_neerest_nei_per_prot_ID = pickle.load(open('data/pickles/ST/dico_neerest_nei_per_prot_ID.data','rb'))
        self.dico_neerest_nei_per_prot_value = pickle.load(open('data/pickles/ST/dico_neerest_nei_per_prot_value.data','rb'))
        self.dico_labels_per_couple = pickle.load(open('data/pickles/ST/dico_labels_per_couple.data', 'rb'))
    
    
    def update_with_extra_task_pairs(self, current_couple, list_train_samples, list_train_labels, value_of_neg_class=0):
        """
        adding extra-task instances to the train set chosen as closer as possible to the tested sample
        """
        print('current couple:', current_couple)
        sys.stdout.flush()
        dico_neerest_pos_nei_per_couple_ID = []
        dico_neerest_neg_nei_per_couple_ID = []

        array_of_sim_ind = np.zeros(len(self.dico_neerest_nei_per_prot_ID[current_couple[0]]), dtype=np.int)
        ## chaque prot de dico_neerest_nei_per_prot_ID est associée à un indice qui le lie à la molécule avec qui il formera le plus couple de plus forte similarité (qu'il n'a pas encore fait -d'où le fait de commencer à l'indice 0 du dico_neerest_nei_per_mol_value)
        array_of_sim_value = np.zeros(len(array_of_sim_ind))
        ## pour chaque prot de dico_neerest_nei_per_prot_ID, on calcul la valeure de similarité de couple la plus forte que chaque prot puisse faire
        for ind in range(len(array_of_sim_ind)):
            array_of_sim_value[ind] = self.dico_neerest_nei_per_prot_value[current_couple[0]][ind] * self.dico_neerest_nei_per_mol_value[current_couple[1]][array_of_sim_ind[ind]]
        sorted_line = np.argsort(array_of_sim_value)


        nb_pos_local = self.PosNei
        nb_neg_local = self.PosNei*self.NegNei
        pos_local = 0
        neg_local = 0

        while pos_local<nb_pos_local or neg_local<nb_neg_local :
            max_ind = np.argmax(array_of_sim_value)
            compared_couple = (self.dico_neerest_nei_per_prot_ID[current_couple[0]][max_ind], self.dico_neerest_nei_per_mol_ID[current_couple[1]][array_of_sim_ind[max_ind]])
#            print(self.dico_labels_per_couple[compared_couple[1] + compared_couple[0]])
#            sys.stdout.flush()
            if compared_couple not in list_train_samples:
                if compared_couple[0] != current_couple[0] and compared_couple[1] != current_couple[1]:
                    if self.condition_on_extra_task(current_couple, compared_couple):
                        if self.dico_labels_per_couple[compared_couple[1] + compared_couple[0]] == 1 and pos_local<nb_pos_local:
                            dico_neerest_pos_nei_per_couple_ID.append(compared_couple)
                            pos_local+=1
                        elif self.dico_labels_per_couple[compared_couple[1] + compared_couple[0]] == value_of_neg_class and neg_local<nb_neg_local:
                            dico_neerest_neg_nei_per_couple_ID.append(compared_couple)
                            neg_local+=1
                            
            array_of_sim_ind[max_ind] += 1
            if array_of_sim_ind[max_ind]==len(self.dico_neerest_nei_per_mol_ID[current_couple[1]]):
                array_of_sim_value[max_ind] = -1
            else:
                array_of_sim_value[max_ind] = self.dico_neerest_nei_per_prot_value[current_couple[0]][max_ind] * self.dico_neerest_nei_per_mol_value[current_couple[1]][array_of_sim_ind[max_ind]]
            if all(i >= len(self.dico_neerest_nei_per_mol_ID[current_couple[1]]) for i in array_of_sim_ind):
                pos_local = nb_pos_local
                neg_local = nb_neg_local
                print('WARNING : all couples were spanned when finding neighboors')
                sys.stdout.flush()

        if current_couple in dico_neerest_pos_nei_per_couple_ID:
                print("current couple in list de neerest nei")
                sys.stdout.flush()
                exit(1)
        elif current_couple in dico_neerest_neg_nei_per_couple_ID:
                print("current couple in list de neerest nei")
                sys.stdout.flush()
                exit(1)

        list_train_samples_local = list_train_samples.copy() + dico_neerest_pos_nei_per_couple_ID + dico_neerest_neg_nei_per_couple_ID
        list_train_labels_local = list_train_labels.copy() + [1 for _ in range(len(dico_neerest_pos_nei_per_couple_ID))] + [value_of_neg_class for _ in range(len(dico_neerest_neg_nei_per_couple_ID))]

        return list_train_samples_local, list_train_labels_local
        
    def condition_on_extra_task(self, current_couple, compared_couple):
        return True
        
        
class NNMT_minus_experiment(NNMT_experiment, ST_minus_experiment):
    """
    negative intra-task pairs are added conditionnaly
    parameters:
    ----------
    type_ST : string, in ['S1','S2','S3','S4'], setting
    NbNeg : int, nb of negative instances in train set is "nb of pos instances in train set * NbNeg"
    PosNei : number of positive extra-task instances in the train set
    NegNei : number of negative extra-task instances in the train set is "NegNei*PosNei"
    centile : int between 0 and 100, centile of the distribution of mol and prot similarity
        Gives a maximum threshold of similarity between the tested sample and samples in the train set
    nb_partial : nb of subdivision of the computation
    """
    
    def __init__(self, type_ST, NbNeg, PosNei, NegNei, centile, nb_partial=50):
        NNMT_experiment.__init__(self,type_ST, NbNeg, PosNei, NegNei)
        self.centile = centile
        self.nb_partial = nb_partial
        self.name_pos = "saved_results/ST/MTNeerestNei_minus_ter_pos_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)+"_Sd:"+str(self.centile)
        self.name_neg = "saved_results/ST/MTNeerestNei_minus_ter_neg_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)+"_Sd:"+str(self.centile)
        
    def load_list_couples(self, ind_sampling, value_of_neg_class):
        return ST_minus_experiment.load_list_couples(self, ind_sampling, value_of_neg_class)        
    
    def condition_on_intra_task(self, Kernel, dico_2indice, compared_sample, tested_sample, threshold):    
        return ST_minus_experiment.condition_on_intra_task(self, Kernel, dico_2indice, compared_sample, tested_sample, threshold)

    def update_with_extra_task_pairs(self, current_couple, list_train_samples, list_train_labels, value_of_neg_class=0):
        return NNMT_experiment.update_with_extra_task_pairs(self, current_couple, list_train_samples, list_train_labels, value_of_neg_class)

    def run(self, ind_partial, ind_sampling):
        return ST_minus_experiment.run(self, ind_partial, ind_sampling)

class NNMT_minusplus_experiment(NNMT_minus_experiment):
    """
    negative intra-task and extra-task pairs are added conditionnaly
    parameters:
    ----------
    type_ST : string, in ['S1','S2','S3','S4'], setting
    NbNeg : int, nb of negative instances in train set is "nb of pos instances in train set * NbNeg"
    PosNei : number of positive extra-task instances in the train set
    NegNei : number of negative extra-task instances in the train set is "NegNei*PosNei"
    centile : int between 0 and 100, centile of the distribution of mol and prot similarity
        Gives a maximum threshold of similarity between the tested sample and samples in the train set
    nb_partial : nb of subdivision of the computation
    """ 
    def __init__(self, type_ST, NbNeg, PosNei, NegNei, centile, nb_partial=50):
        NNMT_minus_experiment.__init__(self,type_ST, NbNeg, PosNei, NegNei, centile, nb_partial)
        self.name_pos = "saved_results/ST/MTNeerestNei_minus+_pos_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)+"_Sd:"+str(self.centile)
        self.name_neg = "saved_results/ST/MTNeerestNei_minus+_neg_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)+"_Sd:"+str(self.centile)
        
    def condition_on_extra_task(self, current_couple, compared_couple):
        return self.K_mol[self.dico_mol2indice[current_couple[1]], self.dico_mol2indice[compared_couple[1]]]<=self.threshold_mol and self.K_prot[self.dico_prot2indice[current_couple[0]], self.dico_prot2indice[compared_couple[0]]]<=self.threshold_prot


if __name__ == '__main__':     
    if 'Kron' in sys.argv[1]:
        type_ST = 'Kron'
    NbNeg = int(sys .argv[2])
    nb_pos = int(sys.argv[3])
    nb_neg = int(sys.argv[4])
    ind_partial = int(sys.argv[5])
    ind_sampling = int(sys.argv[-1])    
    
    if 'minus+' in sys.argv[1]:
        centile = int(sys.argv[6])
        experiment = NNMT_minusplus_experiment(type_ST, NbNeg, nb_pos, nb_neg, centile)
        experiment.run(ind_partial, ind_sampling)            
    elif 'minus' in sys.argv[1]:
        centile = int(sys.argv[6])
        experiment = NNMT_minus_experiment(type_ST, NbNeg, nb_pos, nb_neg, centile)
        experiment.run(ind_partial, ind_sampling)
    else:
        experiment = NNMT_experiment(type_ST, NbNeg, nb_pos, nb_neg)
        experiment.run(ind_partial, ind_sampling)
