from NNMT import *

class Pred_On_Drugbank(NNMT_experiment):
    def __init__(self, type_ST, NbNeg, PosNei, NegNei):
        NNMT_experiment.__init__(self, type_ST, NbNeg, PosNei, NegNei)
        
        self.dico_neerest_nei_per_mol_ID = pickle.load(open('dico_neerest_nei/dico_neerest_nei_per_mol_ID.data','rb'))
        self.dico_neerest_nei_per_mol_value = pickle.load(open('dico_neerest_nei/dico_neerest_nei_per_mol_value.data','rb'))
        self.dico_neerest_nei_per_prot_ID = pickle.load(open('dico_neerest_nei/dico_neerest_nei_per_prot_ID.data','rb'))
        self.dico_neerest_nei_per_prot_value = pickle.load(open('dico_neerest_nei/dico_neerest_nei_per_prot_value.data','rb'))
        
        file_PosDic = "data/datasets/SmallMolMWFilter_UniprotHumanProt_DrugBank_Dictionary.csv"
        file_ProtList = "data/datasets/list_MWFilter_UniprotHumanProt.txt"
        file_MolList = "data/datasets/list_MWFilter_mol.txt"
        self.dico_labels_per_couple = {}
        self.dico_target_of_mol = {}
        self.dico_ligand_of_prot = {}
        self.list_mol = []
        f_in = open(file_PosDic, 'r')
        reader = csv.reader(f_in, delimiter='\t')
        for row in reader:
                self.list_mol.append(row[0])
                self.dico_target_of_mol[row[0]] = [[],[],[]] #1: list_pos; 2: tot list_neg; 3: several list_neg avec meme nbre de neg que dans list_pos
                nb_prot = int(row[4])
                j=0
                while j<nb_prot:
                    self.dico_target_of_mol[row[0]][0].append(row[5+j])
                    if row[5+j] not in self.dico_ligand_of_prot.keys():
                        self.dico_ligand_of_prot[row[5+j]] = [[],[],[]]
                    self.dico_ligand_of_prot[row[5+j]][0].append(row[0])
                    self.dico_labels_per_couple[row[0]+row[5+j]] = 1
                    j+=1
        del reader
        f_in.close()
        for prot in self.dico_ligand_of_prot.keys():
            for cle, valeur in self.dico_target_of_mol.items():
                if prot not in valeur[0]:
                    self.dico_target_of_mol[cle][1].append(prot)
                    self.dico_ligand_of_prot[prot][1].append(cle)
                    self.dico_labels_per_couple[cle+prot] = -1
    
    def make_train(self, couple_test):
        list_train_samples = []
        list_train_labels = []
        list_train_samples_mol, list_train_labels_mol = self.make_ST_train_set(self.dico_target_of_mol[couple_test[1]][0], self.dico_target_of_mol[couple_test[1]][1], [couple_test[0]], self.K_prot, self.dico_prot2indice, self.threshold_prot, -1)
        list_train_samples_prot, list_train_labels_prot = self.make_ST_train_set(self.dico_ligand_of_prot[couple_test[0]][0], self.dico_ligand_of_prot[couple_test[0]][1], [couple_test[1]], self.K_mol, self.dico_mol2indice, self.threshold_mol, -1)

        for el in list_train_samples_mol:
            list_train_samples.append((el,couple_test[1]))
        for el in list_train_samples_prot:
            list_train_samples.append((couple_test[0], el))
        list_train_labels = list_train_labels_mol + list_train_labels_prot
        
        list_train_samples, list_train_labels = self.update_with_extra_task_pairs(couple_test, list_train_samples, list_train_labels, -1)

        K_train, K_test = self.make_Ktrain_and_Ktest_MT(list_train_samples, [couple_test])
        return K_train, K_test, list_train_labels
    
    def run(self,mol):
        dico_pred = {mol:[]}
        for prot in self.dico_target_of_mol[mol][1]:
            print(prot)
            couple_test = (prot,mol)
            K_train, K_test, list_train_labels = self.make_train(couple_test)
            clf = svm.SVC(kernel='precomputed', C=C_value, class_weight='balanced')
            clf.fit(K_train, list_train_labels)
            dico_pred[mol]+=clf.decision_function(K_test).tolist()
            
        pickle.dump(dico_pred, open('tmp/dico_pred_final/'+mol+'.data', 'wb'))
    
    def convert_to_tsv(self,):        
        f_out = open('prediction_on_all_drugbank.tsv','w')
        for mol in list_mol:
            f_out.write(mol+'\tknown_targets\t')
            for prot in dico_target_of_mol[mol][0]:
                f_out.write(prot+'\t')
            f_out.write('unknown_targets\t')
            list_prot = []
            for prot in dico_target_of_mol[mol][1]:
                list_prot.append(prot)
            dico_pred = pickle.load(open('dico_pred_final/'+mol+'.data', 'rb'))
            sorted_pred = np.argsort(dico_pred[mol])[::-1]
            i=0
            for ind in sorted_pred:
                if i<10:
                    f_out.write(list_prot[ind]+':('+str(dico_pred[mol][ind])+')\t')
                i+=1
            f_out.write('\n')
        f_out.close()        
        
if __name__ == '__main__':     
    type_ST = "Kron"
    type_clf = "SVM"
    value_of_neg_class = -1
    type_of_fold = 'S1'
    NbNeg = 10 #2, 5, 10 ou 50
    PosNei = 10
    NegNei = 1
    np.random.seed(769237)
    C_value = 10 
    
    experiment = Pred_On_Drugbank(type_ST, NbNeg, PosNei, NegNei)
    if len(sys.argv)>1:
        mol = sys.argv[1]
        experiment.run(mol)
    else:
        experiment.convert_to_tsv()

