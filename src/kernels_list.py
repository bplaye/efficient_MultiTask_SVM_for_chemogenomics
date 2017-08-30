def make_prot_kernel_list():
    list_prot_kernel_name = []

    list_opening = [1,10,50]
    list_extension = [0.01,0.1,0.5,1,10]
    for opening in list_opening:
        for extension in list_extension:
            list_prot_kernel_name.append("KernelSW_"+str(opening)+'_'+str(extension)+"_normed")

    list_opening = [1, 20,50,100]
    list_extension = [0.01,0.1,1,10]
    list_beta = [0.01, 0.5, 0.05, 0.1, 1]
    for opening in list_opening:
        for extension in list_extension:
            for beta in list_beta:
                list_prot_kernel_name.append("LAkernel_beta_"+str(beta)+"_g_"+str(opening)+"_"+str(extension)+"_ln_normed")

    list_k = [4,5,6,7]
    list_threshold = [6,7.5,9,10.5]
    for k in list_k:
        for t in list_threshold:
            if (k==4 and t!=9 and t!=10.5) or (k==5 and t!=10.5 and t!=6) or (k==6 and t!=6 and t!=7.5 and t!=10.5) or (k==7 and t!=6 and t!=7.5 and t!=9 and t!=10.5):
                list_prot_kernel_name.append("allDrugBankHumanTarget_Profile_CenteredNormalized_k"+str(k)+"_threshold"+str(t))

    list_opening = [100]
    list_extension = [0.01,0.1,0.5,1,10]
    for opening in list_opening:
        for extension in list_extension:
            list_prot_kernel_name.append("KernelSW_"+str(opening)+'_'+str(extension)+"_normed")

    print("len(list_prot_kernel_name)=",len(list_prot_kernel_name))
    return list_prot_kernel_name


##############################################
def make_mol_kernel_list():
    list_mol_kernel_name = []
    
    list_q = [0.05,0.1,0.5]
    list_MI = [2,4,6]
    for q in list_q:
        for MI in list_MI:
            list_mol_kernel_name.append("RandomWalkBasedKernel_q="+str(q)+"_MI="+str(MI))
    list_d = [2,4,6,8]
    for d in list_d:
        list_mol_kernel_name.append("TanimotoMinMax_d="+str(d))
    list_d = [6,8,10,12,14,16]
    for d in list_d:
        list_mol_kernel_name.append("Tanimoto_d="+str(d))
    list_q = [0.01]
    list_MI = [2,4,6]
    for q in list_q:
        for MI in list_MI:
            list_mol_kernel_name.append("RandomWalkBasedKernel_q="+str(q)+"_MI="+str(MI))
    list_d = [18,20]
    for d in list_d:
        list_mol_kernel_name.append("Tanimoto_d="+str(d))

    print("len(list_mol_kernel_name)=",len(list_mol_kernel_name))
    return list_mol_kernel_name
    
    
###############################################
###############################################
###############################################


def make_prot_kernel_list_ClusterCV():
    list_prot_kernel_name = []
    
    list_opening = [1, 20,50,100]
    list_extension = [0.01,0.1,1,10]
    list_beta = [0.01, 0.5, 0.05, 0.1, 1]
    for opening in list_opening:
        for extension in list_extension:
            for beta in list_beta:
                list_prot_kernel_name.append("LAkernel_beta_"+str(beta)+"_g_"+str(opening)+"_"+str(extension)+"_ln_normed")

    list_k = [4,5,6,7]
    list_threshold = [6,7.5,9,10.5]
    for k in list_k:
        for t in list_threshold:
            if (k==4 and t!=9 and t!=10.5) or (k==5 and t!=10.5 and t!=6) or (k==6 and t!=6 and t!=7.5 and t!=10.5) or (k==7 and t!=6 and t!=7.5 and t!=9 and t!=10.5):
                list_prot_kernel_name.append("allDrugBankHumanTarget_Profile_CenteredNormalized_k"+str(k)+"_threshold"+str(t))

    print("len(list_prot_kernel_name)=",len(list_prot_kernel_name))
    return list_prot_kernel_name


##############################################
def make_mol_kernel_list_ClusterCV():
    list_mol_kernel_name = []
    
    list_d = [6,8,10,12,14,16,18]
    for d in list_d:
        list_mol_kernel_name.append("Tanimoto_d="+str(d))
        
    print("len(list_mol_kernel_name)=",len(list_mol_kernel_name))
    return list_mol_kernel_name
    
    
    
