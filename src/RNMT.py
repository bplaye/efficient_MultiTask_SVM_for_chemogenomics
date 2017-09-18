from src.ST import *


class RNMT_experiment(ST_experiment):
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
        self.name = 'saved_results/ST/MTRandomNei_'+self.type_ST+'_S1_'+self.type_clf+'_NbNeg='+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)
        self.PosNei = PosNei
        self.NegNei = NegNei
        self.list_pos_couples = pickle.load(open('data/pickles/ST/list_pos_couples.data', 'rb'))
        self.list_neg_couples = pickle.load(open('data/pickles/ST/list_neg_couples.data', 'rb'))
        self.dico_labels_per_couple = pickle.load(open('data/pickles/ST/dico_labels_per_couple.data', 'rb'))
    
    def update_with_extra_task_pairs(self, current_couple, list_train_samples, list_train_labels, value_of_neg_class=0):
        """
        adding extra-task instances to the train set chosen randomly
        """
        random_pos_nei = []
        random_neg_nei = []

        nb_pos_local = self.PosNei
        nb_neg_local = self.PosNei*self.NegNei
        pos_local = 0
        neg_local = 0
        
        pos_rand_list_of_indices = [i for i in range(len(self.list_pos_couples))]
        np.random.shuffle(pos_rand_list_of_indices)
        for rand_int in pos_rand_list_of_indices:
            if self.list_pos_couples[rand_int] not in list_train_samples and self.list_pos_couples[rand_int][0]!=current_couple[0] and self.list_pos_couples[rand_int][1]!=current_couple[1] and pos_local<nb_pos_local:
                if self.condition_on_extra_task(current_couple, self.list_pos_couples[rand_int]):
                    random_pos_nei.append(self.list_pos_couples[rand_int])
                    pos_local+=1
            elif pos_local>=nb_pos_local:
                break
        
        neg_rand_list_of_indices = [i for i in range(len(self.list_neg_couples))]
        np.random.shuffle(neg_rand_list_of_indices)
        for rand_int in neg_rand_list_of_indices:
            if self.list_neg_couples[rand_int] not in list_train_samples and self.list_neg_couples[rand_int][0]!=current_couple[0] and self.list_neg_couples[rand_int][1]!=current_couple[1] and neg_local<nb_neg_local:
                if self.condition_on_extra_task(current_couple, self.list_neg_couples[rand_int]):
                    random_neg_nei.append(self.list_neg_couples[rand_int])
                    neg_local+=1
            elif neg_local>=nb_neg_local:
                break
                
        #print(random_neg_nei)
        #print(random_pos_nei)
        list_train_samples_local = list_train_samples.copy() + random_pos_nei + random_neg_nei
        list_train_labels_local = list_train_labels.copy() + [1 for _ in range(len(random_pos_nei))] + [value_of_neg_class for _ in range(len(random_neg_nei))]

        return list_train_samples_local, list_train_labels_local
        
    def condition_on_extra_task(self, current_couple, compared_couple):
        return True
        
        
class RNMT_minus_experiment(RNMT_experiment, ST_minus_experiment):
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
    
    def __init__(self, type_ST, NbNeg, PosNei, NegNei, centile, nb_partial=20):
        RNMT_experiment.__init__(self,type_ST, NbNeg, PosNei, NegNei)
        self.centile = centile
        self.nb_partial = nb_partial
        self.name_pos = "saved_results/ST/MTRandomNei_minus_ter_pos_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)+"_Sd:"+str(self.centile)
        self.name_neg = "saved_results/ST/MTRandomNei_minus_ter_neg_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)+"_Sd:"+str(self.centile)
        
    def load_list_couples(self, ind_sampling, value_of_neg_class):
        return ST_minus_experiment.load_list_couples(self, ind_sampling, value_of_neg_class)        
    
    def update_with_extra_task_pairs(self, current_couple, list_train_samples, list_train_labels, value_of_neg_class=0):
        return RNMT_experiment.update_with_extra_task_pairs(self, current_couple, list_train_samples, list_train_labels, value_of_neg_class)

    def condition_on_intra_task(self, Kernel, dico_2indice, compared_sample, tested_sample, threshold):    
        return ST_minus_experiment.condition_on_intra_task(self, Kernel, dico_2indice, compared_sample, tested_sample, threshold)

    def run(self, ind_partial, ind_sampling):
        return ST_minus_experiment.run(self, ind_partial, ind_sampling)

class RNMT_minusplus_experiment(RNMT_minus_experiment):
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
    def __init__(self, type_ST, NbNeg, PosNei, NegNei, centile, nb_partial=20):
        RNMT_minus_experiment.__init__(self,type_ST, NbNeg, PosNei, NegNei, centile, nb_partial)
        self.name_pos = "saved_results/ST/MTRandomNei_minus+_pos_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)+"_Sd:"+str(self.centile)
        self.name_neg = "saved_results/ST/MTRandomNei_minus+_neg_"+self.type_ST+"_S1_"+self.type_clf+"_NbNeg="+str(self.NbNeg)+'_PosNei'+str(PosNei)+'_NegNei='+str(NegNei)+"_Sd:"+str(self.centile)
        
    def condition_on_extra_task(self, current_couple, compared_couple):
        return self.K_mol[self.dico_mol2indice[current_couple[1]], self.dico_mol2indice[compared_couple[1]]]<=self.threshold_mol and self.K_prot[self.dico_prot2indice[current_couple[0]], self.dico_prot2indice[compared_couple[0]]]<=self.threshold_prot


if __name__ == '__main__':     
    if 'Kron' in sys.argv[1]:
        type_ST = 'Kron'
    NbNeg = int(sys.argv[2])
    nb_pos = int(sys.argv[3])
    nb_neg = int(sys.argv[4])
    ind_partial = int(sys.argv[5])
    ind_sampling = int(sys.argv[-1])    
    
    if 'minus+' in sys.argv[1]:
        centile = int(sys.argv[6])
        experiment = RNMT_minusplus_experiment(type_ST, NbNeg, nb_pos, nb_neg, centile)
        experiment.run(ind_partial, ind_sampling)            
    elif 'minus' in sys.argv[1]:
        centile = int(sys.argv[6])
        experiment = RNMT_minus_experiment(type_ST, NbNeg, nb_pos, nb_neg, centile)
        experiment.run(ind_partial, ind_sampling)
    else:
        experiment = RNMT_experiment(type_ST, NbNeg, nb_pos, nb_neg)
        experiment.run(ind_partial, ind_sampling)
