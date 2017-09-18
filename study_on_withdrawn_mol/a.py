import csv

dico_mol = {}
list_mol = []
reader = csv.reader(open('wd.tsv', 'r', encoding='latin'), delimiter='\t')
for row in reader:
    name = ''
    for l in row[0]:
        if l==' ' or l=='(' or l=='Ê':
            break
        else:
            name+=l
    dico_mol[name] = {'name':name, 'year':row[1].replace("Ê", " "), 'country':row[2].replace("Ê", " "), 'DBid':''}
    list_mol.append(name)
del reader

reader = csv.reader(open('wd_wDB.tsv', 'r', encoding='latin'), delimiter='\t')
for row in reader:
    if row[0] in dico_mol.keys():
        dico_mol[row[0]]['DBid'] = row[1].replace("Ê", "")
    else:
        print(row[0]+' not in dico_mol : wd_wDB')
del reader

list_considered_mol = []
reader = csv.reader(open('wd_consideredmol.tsv', 'r', encoding='latin'), delimiter='\t')
for row in reader:
    if row[0] in dico_mol.keys():
        list_considered_mol.append(row[0])
    else:
        print(row[0]+' not in dico_mol : wd_consideredmol')
del reader

f_out = open('t.txt', 'w')
f_out.write('\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|} \n \\hline\n ')
f_out.write(' name & year & country & DrugBank id \\\\ \\hline\n')
for mol in list_mol:
    f_out.write(dico_mol[mol]['name']+' & '+dico_mol[mol]['year']+' & '+dico_mol[mol]['country']+' & '+dico_mol[mol]['DBid']+'\\\\ \\hline\n')
f_out.write('\end{tabular}\n\\caption{List of withdrawn drugs}\n\\label{table:wd_table}\n\\end{table}')


f_out = open('tt.txt', 'w')
f_out.write('\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|} \n \\hline\n ')
f_out.write('name & year & country & DrugBank id \\\\ \\hline\n')
for mol in list_considered_mol:
    f_out.write(dico_mol[mol]['name']+' & '+dico_mol[mol]['year']+' & '+dico_mol[mol]['country']+' & '+dico_mol[mol]['DBid']+'\\\\ \\hline\n')
f_out.write('\end{tabular}\n\\caption{List of considered withdrawn drugs}\n\\label{table:consideredwd_table}\n\\end{table}')
