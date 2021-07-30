#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Evluation of RETECS Vs DeepOrder using NAPFD
import pprint
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

PARALLEL = True
VISUALIZE_RESULTS = False
def save_figures(fig, filename):

    FIGURE_DIR = os.path.abspath(os.getcwd())
    fig.savefig(os.path.join(FIGURE_DIR, filename + '.pdf'), bbox_inches='tight')


def figsize_column(scale, height_ratio=1.0):
    fig_width_pt = 240  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

def figsize_text(scale, height_ratio=1.0):
    fig_width_pt = 504  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 20,
        }

font2 = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 18,
        }

reward_names = {
    'failcount': 'Failure Count Reward',
    'tcfail': 'Test Case Failure Reward',
    'timerank': 'Time-ranked Reward'
}

env_names = {
    'paintcontrol': 'ABB Paint Control',
    'iofrol': 'ABB IOF/ROL',
    'gsdtsr': 'GSDTSR',
    'cisco' : 'Cisco'
}


flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
flatui2 = ["#3498db", "#e74c3c"]
NAPFD_RETECS_1 = []


def visualize():
    search_pattern = 'rq_*_stats.p'

    mean_df = pd.read_csv('mean_df.csv', sep=";")

# Cisco and IOF/ROL results for DeepOrder and RETECS
df_output = pd.read_csv('output_2.csv', sep=";")


fig, axarr = plt.subplots(1, 2, sharey=True, sharex=True, figsize=figsize_text(2.3, 0.5))
plotname = 'rq1_napfd_1'
FIGURE_DIR = ''
subplot_labels = ['(a)']
row =0
for column, env in enumerate(sorted(df_output['env'].unique(), reverse=True)):
    #for row, rewardfun in enumerate(df_output['rewardfun'].unique()):
        for agidx, (labeltext, agent, linestyle) in enumerate(
                    [('RETECS Network', 'mlpclassifier', '-'), ('DeepOrder', 'DeepOrder', '-')]):
            rel_df = df_output[(df_output['env'] == env)]
            #print (rel_df)

            rel_df[rel_df['agent'] == agent].plot(x='step', y='NAPFD', label=labeltext, fontsize=14,ylim=[0, 1], linewidth=1.6,
                                                  style=linestyle, color=sns.color_palette(flatui)[agidx], ax=axarr[column])

            x = rel_df.loc[rel_df['agent'] == agent, 'step']
            y = rel_df.loc[rel_df['agent'] == agent, 'NAPFD']



            #print (rel_df_2.columns)
            #rel_df_1.to_csv('rel_df_2'+'.csv', mode= 'a',sep=";",index = False)
            trend = np.poly1d(np.polyfit(x, y, 1))
            axarr[column].plot(x, trend(x), linestyle, color=sns.color_palette(flatui)[agidx], linewidth=3.5)
            
            axarr[column].legend_.remove()

        axarr[column].set_xticks(np.arange(0, 350, 30), minor=False)
        axarr[column].set_xticklabels([0, '', 50, '', 100, '', 150, '', 200], minor=False,fontsize=14)
        axarr[column].xaxis.grid(True, which='major')
        axarr[column].set_xlabel('Time Window',fontsize=17)
        axarr[column].set_ylabel('NAPFD',fontsize=17)
        axarr[column].set_title(env_names[env] , fontdict=font2)


        if column == 1:
            axarr[column].legend(loc=2, ncol=1, frameon=True, bbox_to_anchor=(0.6, 1.03),fontsize=14)
        
fig.subplots_adjust(wspace=0.06, hspace=0.3)
save_figures(fig, plotname)
plt.clf()


##------------------------------------------------------------------------##

# Paint Control and GSDTSR Results for DeepOrder and RETECS

df_output = pd.read_csv('output_3.csv', sep=";")


fig, axarr = plt.subplots(1, 2, sharey=True, sharex=True, figsize=figsize_text(2.3, 0.5))
plotname = 'rq1_napfd_2'
FIGURE_DIR = ''
subplot_labels = ['(a)']
row =0
for column, env in enumerate(sorted(df_output['env'].unique(), reverse=True)):
    #for row, rewardfun in enumerate(df_output['rewardfun'].unique()):
        for agidx, (labeltext, agent, linestyle) in enumerate(
                    [('RETECS Network', 'mlpclassifier', '-'), ('DeepOrder', 'DeepOrder', '-')]):
            rel_df = df_output[(df_output['env'] == env)]
            #print (rel_df)

            rel_df[rel_df['agent'] == agent].plot(x='step', y='NAPFD', label=labeltext, fontsize=14,ylim=[0, 1], linewidth=1.6,
                                                  style=linestyle, color=sns.color_palette(flatui)[agidx], ax=axarr[column])

            x = rel_df.loc[rel_df['agent'] == agent, 'step']
            y = rel_df.loc[rel_df['agent'] == agent, 'NAPFD']



            #print (rel_df_2.columns)
            #rel_df_1.to_csv('rel_df_2'+'.csv', mode= 'a',sep=";",index = False)
            trend = np.poly1d(np.polyfit(x, y, 1))
            axarr[column].plot(x, trend(x), linestyle, color=sns.color_palette(flatui)[agidx], linewidth=3.5)
            
            axarr[column].legend_.remove()

        axarr[column].set_xticks(np.arange(0, 350, 30), minor=False)
        axarr[column].set_xticklabels([0, '', 50, '', 100, '', 150, '', 200, '', 300], minor=False,fontsize=14)
        axarr[column].xaxis.grid(True, which='major')
        axarr[column].set_xlabel('Time Window',fontsize=17)
        axarr[column].set_ylabel('NAPFD',fontsize=17)
        axarr[column].set_title(env_names[env] , fontdict=font2)


        if column == 1:
            axarr[column].legend(loc=2, ncol=1, frameon=True, bbox_to_anchor=(0.6, 1.03),fontsize=14)
        
fig.subplots_adjust(wspace=0.06, hspace=0.3)
save_figures(fig, plotname)
plt.clf()




if __name__ == '__main__':

    if VISUALIZE_RESULTS:
        visualize()
