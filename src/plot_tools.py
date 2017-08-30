import csv
import numpy as np
import collections
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
import statistics
import math
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.pyplot as plt

CBcdict={
    'Bl':(0,0,0),
    'Or':(.9,.6,0),
    'SB':(.35,.7,.9),
    'bG':(0,.6,.5),
    'Ye':(.95,.9,.25),
    'Bu':(0,.45,.7),
    'Ve':(.8,.4,0),
    'rP':(.8,.6,.7),
}

##Single color gradient maps
def lighter(colors):
    li=lambda x: x+.5*(1-x)
    return (li(colors[0]),li(colors[1]),li(colors[2]))

def darker(colors):
    return (.5*colors[0],.5*colors[1],.5*colors[2])

CBLDcm={}
for key in CBcdict:
        CBLDcm[key]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key,[lighter(CBcdict[key]),darker(CBcdict[key])])


##Two color gradient maps
CB2cm={}
for key in CBcdict:
    for key2 in CBcdict:
        if key!=key2: CB2cm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],CBcdict[key2]])

##Two color gradient maps with white in the middle
CBWcm={}
for key in CBcdict:
    for key2 in CBcdict:
        if key!=key2: CBWcm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],(1,1,1),CBcdict[key2]])

##Two color gradient maps with Black in the middle
CBBcm={}
for key in CBcdict:
    for key2 in CBcdict:
        if key!=key2: CBBcm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],(0,0,0),CBcdict[key2]])

##Change default color cycle
matplotlib.rcParams['axes.color_cycle'] = [CBcdict[c] for c in sorted(CBcdict.keys())]




def plot_score_histo(title_local, list_score1, list_score2, list_score1_stdev, list_score2_stdev, score1_name, score2_name, xlabel_string, list_x_label, plot_option, **kwargs):
    """
    plots 2 scores side by side with a barplot

    parameters:
    -----------
    title_local : string, title of the plot
    list_score1 : list of real values, list of scores of type 1
    list_score2 : list of real values, list of scores of type 2
    list_score1_stdev : list of real values, list of standard deviation of scores of type 1
    list_score2_stdev : list of real values, list of standard deviation of scores of type 2
    score1_name : string, name of the score of type 1 (in our case ROC-AUC)
    score2_name : string, name of the score of type 2 (in our case AUPR)
    xlabel_string : string, legend of the x-axis
    list_x_label : list of string, labels on the x-axis
    plot_option : string ('print', 'save' or 'print_and_save') 
    **kwargs :  contains the paths when saving plots
    """
    FONTSIZE = 20
    
    pp_local = kwargs.get('pp', None)
    f_out_local = kwargs.get('f_out', None)
    
    cmap = CB2cm['BuVe']
    colors = [cmap(i) for i in np.linspace(0, 1, 2)]
    
    for i_temp, x_label in enumerate(list_x_label):
        if x_label[:2]=='MT':
            list_x_label[i_temp] = x_label[3:]
            if list_x_label[i_temp]=='Nested5foldCV':
                 list_x_label[i_temp] = 'Nested-5foldCV'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title_local)
    width = 0.35
    list_x = []
    list_x_bis = []
    for i in range(len(list_score1)):
        list_x.append(i)
        list_x_bis.append(i+width)
    ## the bars
    print(list_x)
    print(list_score1)
    print(list_score1_stdev)
    rects1 = ax.bar(list_x, list_score1, width, color=colors[0], yerr=list_score1_stdev)
    rects2 = ax.bar(list_x_bis, list_score2, width, color=colors[1], yerr=list_score2_stdev)
    # axes and labels
    ax.set_xlim(min(list_x)-width,max(list_x_bis)+width)
    ax.set_ylim(min(min(list_score1),min(list_score2))-0.05,1)
    plt.yticks(fontsize=FONTSIZE)
    #ax.set_ylim(0,45)
    ax.set_ylabel('Scores', fontsize=FONTSIZE)
    ax.set_xlabel(xlabel_string, fontsize=FONTSIZE)
    xTickMarks = list_x_label
    ax.set_xticks([(list_x[i] + list_x_bis[i])/2 for i in range(len(list_x))])
    xtickNames = ax.set_xticklabels(xTickMarks)
    #plt.setp(xtickNames, rotation=45, fontsize=10)
    plt.setp(xtickNames, fontsize=FONTSIZE)
    ## add a legend
    lgd = ax.legend( (rects1[0], rects2[0]), (score1_name, score2_name), loc="upper right")
    for label in lgd.get_texts():
        label.set_fontsize(FONTSIZE)
    lgd.get_title().set_fontsize(FONTSIZE)

    if plot_option=="save" or plot_option=="print_and_save":
        f_out_local.write(title_local+':\n')
        for ind in range(len(list_x_label)):
            f_out_local.write(list_x_label[ind]+'\t'+str(list_score1[ind])+'\t'+str(list_score1_stdev[ind])+'\t'+str(list_score2[ind])+'\t'+str(list_score2_stdev[ind])+'\n')

    return lgd, xtickNames









def plot_score_curve(title_local, list_of_list_of_score, list_of_list_of_score_stdev, list_x_label, score_name, xlabel_string, list_x_names, title_legend, plot_option, **kwargs):
    """
    plots a list of curves ith error bars

    parameters:
    -----------
    title_local : string, title of the plot
    list_of_list_of_score : list of list of real values, each sublist is a list of scores
    list_of_list_of_score_stdev : list of list of real values, leach sublist is a list of stadard deviation of the score
    list_x_label : list of real values, labels on the x-axis
    score_name : string, name of the score(in our case AUPR or ROC-AUC)
    xlabel_string : string, legend of the x-axis
    list_x_names : list of strings, list of names associated to the different plots
    title_legend: string, title of the legend
    plot_option : string ('print', 'save' or 'print_and_save') 
    **kwargs :  contains the paths when saving plots
    """
    FONTSIZE = 20
    
    pp_local = kwargs.get('pp', None)
    f_out_local = kwargs.get('f_out', None)  
    
    list_x_label_temp = []
    for label in list_x_label:
        if label!="full":
            list_x_label_temp.append(int(label))
        else:
            list_x_label_temp.append(100)
            
    cmap = CB2cm['BuVe']
    list_color = [cmap(i) for i in np.linspace(0, 1, len(list_of_list_of_score))]

    plt.figure()
    plt.xlabel(xlabel_string, fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.ylabel(score_name, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.title(title_local)
    min_y = 1
    for list_score in list_of_list_of_score:
        print(list_score)
        min_local = min(list_score)
        if min_local<min_y:
            min_y = min_local
    if max(list_x_label_temp)<100:
        plt.xlim([min(list_x_label_temp)-1,max(list_x_label_temp)+1])
    else:
        plt.xlim([min(list_x_label_temp)-5,max(list_x_label_temp)+5])
    plt.ylim([min_y-0.05,1])
    
    # mastering xticks : plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    #print(list_x_label_temp)
    #print(list_of_list_of_score)
    #print(list_of_list_of_score_stdev)
    if plot_option=="save" or plot_option=="print_and_save":
        f_out_local.write('\t')
        for i_el, el in enumerate(list_x_label_temp):
            f_out_local.write(str(el)+' & ')
        f_out_local.write('\\ \n \hline \n')
    
    for ind_local, list_score in enumerate(list_of_list_of_score):
        if plot_option=="save" or plot_option=="print_and_save":
            f_out_local.write(list_x_names[ind_local]+' & ')
            for i_el, el in enumerate(list_x_label_temp):
                f_out_local.write('AUPR='+str(round(100*list_score[i_el],2))+', std='+str(round(100*list_of_list_of_score_stdev[ind_local][i_el],2))+' & ')
            f_out_local.write('\\\\ \n \hline \n')
        
        plt.errorbar(list_x_label_temp, list_score, list_of_list_of_score_stdev[ind_local], color=list_color[ind_local], label=list_x_names[ind_local], linewidth=3.0,elinewidth=3,
    markeredgewidth=3)
    # Now add the legend with some customizations.
    legend = plt.legend(loc='center left', title=title_legend, bbox_to_anchor=(1,0.5))

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    #frame = legend.get_frame()
    #frame.set_facecolor('0.90')

    # Set the fontsize
    if len(list_x_names)>1:
        for label in legend.get_texts():
            label.set_fontsize(FONTSIZE)
        legend.get_title().set_fontsize(FONTSIZE)

    #for label in legend.get_lines():
    #    label.set_linewidth(1.5)  # the legend line width
    
    return legend
    

