from src.NNMT import *
from src.RNMT import *

class NNMT_family_experiment(NNMT_experiment):
    """
    compute LOOCV performances of NNMT for a specific "family and type_kernel" and specific "type_ST, NbNeg, PosNei, NegNei"
    parameters:
    ----------
    type_ST : string, in ['S1','S2','S3','S4'], setting
    NbNeg : int, nb of negative instances in train set is "nb of pos instances in train set * NbNeg"
    PosNei : number of positive extra-task instances in the train set
    NegNei : number of negative extra-task instances in the train set is "NegNei*PosNei"
    family : string, 'GPCR' or 'kinase' or 'IC'
    type_kernel : string, 'TotKernel' or 'FamilyKernel', use kernels on sequences or on the hierarchy of the family
    """
    def __init__(self, type_ST, NbNeg, PosNei, NegNei, family, type_kernel):
        NNMT_experiment.__init__(self, type_ST, NbNeg, PosNei, NegNei)
        self.centile = 0
        self.difficulty = 'normal'
        self.name = 'saved_results/family/MultiFamilyNeerestStudy:'+family+'_'+type_kernel+'_Intra_'+self.type_ST+'_S1_'+self.type_clf+'_NbNeg='+str(self.NbNeg)+'_PosNei='+str(self.PosNei)+'_NegNei='+str(self.NegNei)+'_'+self.difficulty+':'+str(self.centile)
        self.family = family
        self.type_kernel = type_kernel
        mol_kernel_name = 'Tanimoto_d=8'
        if type_kernel=="TotKernel":
            prot_kernel_name = 'allDrugBankHumanTarget_Profile_CenteredNormalized_k5_threshold7.5'
            self.dico_neerest_nei_per_mol_ID = pickle.load(open('data/pickles/family/dico_family_neerest_nei_per_mol_ID'+family+'.data', 'rb'))
            self.dico_neerest_nei_per_mol_value = pickle.load(open('data/pickles/family/dico_family_neerest_nei_per_mol_value'+family+'.data', 'rb'))
            self.dico_neerest_nei_per_prot_ID = pickle.load(open('data/pickles/family/dico_family_neerest_nei_per_prot_ID'+family+'.data', 'rb'))
            self.dico_neerest_nei_per_prot_value = pickle.load(open('data/pickles/family/dico_family_neerest_nei_per_prot_value'+family+'.data', 'rb'))
        elif type_kernel=="FamilyKernel":
            prot_kernel_name = family+'_Kernel'
            self.dico_neerest_nei_per_mol_ID = pickle.load(open('data/pickles/family/dico_family_neerest_nei_per_mol_ID'+family+'.data', 'rb'))
            self.dico_neerest_nei_per_mol_value = pickle.load(open('data/pickles/family/dico_family_neerest_nei_per_mol_value'+family+'.data', 'rb'))
            self.dico_neerest_nei_per_prot_ID = pickle.load(open('dico_neerest_nei/dico_neerest_nei_per_prot_'+family+'_ID.data', 'rb'))
            self.dico_neerest_nei_per_prot_value = pickle.load(open('dico_neerest_nei/dico_neerest_nei_per_prot_'+family+'_value.data', 'rb'))
        dico_labels_per_couple = pickle.load(open('data/pickles/family/dico_labels_per_couple'+family+'.data', 'rb'))
        
        self.dico_target_of_mol = pickle.load(open('data/pickles/family/dico_family_target_of_mol'+family+'.data', 'rb'))
        self.dico_ligand_of_prot = pickle.load(open('data/pickles/family/dico_family_ligand_of_prot'+family+'.data', 'rb'))
        self.K_prot = pickle.load(open('data/kernels/'+prot_kernel_name+'.data', 'rb'))
        self.dico_prot2indice = pickle.load(open('data/kernels/dico_2indice_'+prot_kernel_name+'.data', 'rb'))
        self.dico_indice2prot = pickle.load(open('data/kernels/dico_indice2_'+prot_kernel_name+'.data', 'rb'))
        
        self.K_mol = pickle.load(open('data/kernels/'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
        self.dico_indice2mol = pickle.load(open('data/kernels/dico_indice2_'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
        self.dico_mol2indice = pickle.load(open('data/kernels/dico_2indice_'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
            
    def load_list_couples(self, ind_sampling, value_of_neg_class):
        list_pos_test_couples = pickle.load(open('data/pickles/family/list_pos_test_couples'+self.family+'.data', 'rb'))
        list_partial_neg_test_couples = pickle.load(open('data/pickles/family/list_partial_neg_test_couples'+self.family+'.data', 'rb'))
        list_couples_tot = list_pos_test_couples + list_partial_neg_test_couples[ind_sampling]
        list_labels_tot = [1 for _ in range(len(list_pos_test_couples))] + [-1 for _ in range(len(list_partial_neg_test_couples[ind_sampling]))]
        return list_couples_tot, list_labels_tot




        
class RNMT_family_experiment(RNMT_experiment):
    """
    compute LOOCV performances of RNMT for a specific "family and type_kernel" and specific "type_ST, NbNeg, PosNei, NegNei"
    parameters:
    ----------
    type_ST : string, in ['S1','S2','S3','S4'], setting
    NbNeg : int, nb of negative instances in train set is "nb of pos instances in train set * NbNeg"
    PosNei : number of positive extra-task instances in the train set
    NegNei : number of negative extra-task instances in the train set is "NegNei*PosNei"
    family : string, 'GPCR' or 'kinase' or 'IC'
    type_kernel : string, 'TotKernel' or 'FamilyKernel', use kernels on sequences or on the hierarchy of the family
    """
    def __init__(self, type_ST, NbNeg, PosNei, NegNei, family, type_kernel):
        RNMT_experiment.__init__(self, type_ST, NbNeg, PosNei, NegNei)
        self.centile = 0
        self.difficulty = 'normal'
        self.name = 'saved_results/family/MultiFamilyRandomStudy:'+family+'_'+type_kernel+'_Intra_'+self.type_ST+'_S1_'+self.type_clf+'_NbNeg='+str(self.NbNeg)+'_PosNei='+str(self.PosNei)+'_NegNei='+str(self.NegNei)+'_'+self.difficulty+':'+str(self.centile)
        self.family = family
        self.type_kernel = type_kernel
        
        self.list_pos_couples = pickle.load(open('data/pickles/family/list_pos_couples'+family+'.data', 'rb'))
        self.list_neg_couples = pickle.load(open('data/pickles/family/list_neg_couples'+family+'.data', 'rb'))
        self.dico_labels_per_couple = dico_labels_per_couple = pickle.load(open('data/pickles/family/dico_labels_per_couple'+family+'.data', 'rb'))
        
        mol_kernel_name = 'Tanimoto_d=8'
        if type_kernel=="TotKernel":
            prot_kernel_name = 'allDrugBankHumanTarget_Profile_CenteredNormalized_k5_threshold7.5'
        elif type_kernel=="FamilyKernel":
            prot_kernel_name = family+'_Kernel'
        dico_labels_per_couple = pickle.load(open('data/pickles/family/dico_labels_per_couple'+family+'.data', 'rb'))
        self.dico_target_of_mol = pickle.load(open('data/pickles/family/dico_family_target_of_mol'+family+'.data', 'rb'))
        self.dico_ligand_of_prot = pickle.load(open('data/pickles/family/dico_family_ligand_of_prot'+family+'.data', 'rb'))
        self.K_prot = pickle.load(open('data/kernels/'+prot_kernel_name+'.data', 'rb'))
        self.dico_prot2indice = pickle.load(open('data/kernels/dico_2indice_'+prot_kernel_name+'.data', 'rb'))
        self.dico_indice2prot = pickle.load(open('data/kernels/dico_indice2_'+prot_kernel_name+'.data', 'rb'))
        self.K_mol = pickle.load(open('data/kernels/'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
        self.dico_indice2mol = pickle.load(open('data/kernels/dico_indice2_'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
        self.dico_mol2indice = pickle.load(open('data/kernels/dico_2indice_'+mol_kernel_name+'_DrugBankSmallMolMWFilterHuman.data', 'rb'))
        
    def load_list_couples(self, ind_sampling, value_of_neg_class):
        list_pos_test_couples = pickle.load(open('data/pickles/family/list_pos_test_couples'+self.family+'.data', 'rb'))
        list_partial_neg_test_couples = pickle.load(open('data/pickles/family/list_partial_neg_test_couples'+self.family+'.data', 'rb'))
        list_couples_tot = list_pos_test_couples + list_partial_neg_test_couples[ind_sampling]
        list_labels_tot = [1 for _ in range(len(list_pos_test_couples))] + [-1 for _ in range(len(list_partial_neg_test_couples[ind_sampling]))]
        return list_couples_tot, list_labels_tot
        
        
        
if __name__ == '__main__':     
    if 'Kron' in sys.argv[1]:
        type_ST = 'Kron'
    NbNeg = int(sys.argv[2])
    PosNei = int(sys.argv[3])
    NegNei = int(sys.argv[4])
    family = sys.argv[5]
    type_kernel = sys.argv[6]
    ind_partial = int(sys.argv[7])
    ind_sampling = int(sys.argv[-1])
    
    if 'NNMT' in sys.argv[1]:
        experiment = NNMT_family_experiment(type_ST, NbNeg, PosNei, NegNei, family, type_kernel)
        experiment.run(ind_partial, ind_sampling) 
    elif'RNMT' in sys.argv[1]:
        experiment = RNMT_family_experiment(type_ST, NbNeg, PosNei, NegNei, family, type_kernel)
        experiment.run(ind_partial, ind_sampling) 
    
