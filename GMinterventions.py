#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:47:05 2020

@author: raechen
"""
import pandas as pd
import _root as r
import sys
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


data = r.GetGMData()   
# data = r.GetRegionData('London')

timeline = pd.read_excel(r"interventions_timeline.xlsx", 
                         index_col='Date', sheet_name='Actions',
                         encoding=sys.getfilesystemencoding())

by = 'NewCases'
district = 'GM'
shift = 7

r.plotInterventions(data, timeline, by, district, shift=shift)

local_data = r.AppendVariables(data, timeline, district, scenario=2, shift=shift)

diff, paras, fitting = r.InterventionFitting(local_data, timeline, 'exp', by, district, 
                                             shift=shift, smooth=24, degree=1, last=15, 
                                             reset=True, tolerance=3000)

# diff, paras = InterventionFitting(local_data, 'log', by=by, last=15)
#diff.to_excel(r"GMResults.xlsx", sheet_name=district)


'''
################################## Comparison ###################################
'''
future = len(local_data['NewCases'][local_data['Action 11']==1].values)
initial_result = local_data['NewCases'][local_data['Action 11']==1].values[0]

parameters = [paras['Action 1'], paras['Action 9'], paras['Action 11']]
labels = ['Lockdown I', 'Lockdown II + school open', 'Lockdown 3 + Vacc']


#curve plot
plt.figure(1, figsize=(8,6))
plt.axvline(x=0, linestyle=':',
            label='Start Point', linewidth=3, color='r')

plt.plot(local_data['NewCases'][local_data['Action 1']==1].values[:future],
         color=sns.color_palette()[0], alpha=.2, linewidth=2)
plt.plot(fitting['NewCases'][fitting['Action 1']==1].values[:future],
         color=sns.color_palette()[0], linewidth=3, label=labels[0], alpha=.3)

plt.plot(local_data['NewCases'][local_data['Action 9']==1].values[:future], 
         color=sns.color_palette()[1], alpha=.2, linewidth=2)
plt.plot(fitting['NewCases'][fitting['Action 9']==1].values[:future],
          color=sns.color_palette()[1], linewidth=3, label=labels[1])

plt.plot(local_data['NewCases'][local_data['Action 11']==1].values[:future], 
         color=sns.color_palette()[2], alpha=.2, linewidth=2)
plt.plot(fitting['NewCases'][fitting['Action 11']==1].values[:future],
          color=sns.color_palette()[2], linewidth=3, label=labels[2])

plt.legend()
plt.xlabel('Days since the action')
plt.ylabel('Cases')
plt.ylim([0,12500])
plt.title('Lockdown I, II and III: curve comparison')


#trend plot
plt.figure(2, figsize=(8,6))
plt.axvline(x=0, linestyle=':',
            label='Start Point', linewidth=3, color='r')

for j in range(len(parameters)):
    cases = []  
    for i in range(future):
        value = np.exp(parameters[j][0]*i)*initial_result
        cases.append(value if value >=0.1 else 0)
     
    if j == 0:
        alpha = .3
    else:
        alpha = 1
    
    plt.plot(cases, linewidth=3, label=labels[j], alpha=alpha)

plt.title('Lockdown I, II and III: trend fitting')
plt.xlabel('Days since the action')
plt.ylabel('Cases')
plt.ylim([0,12500])
plt.grid(True)
plt.legend()



'''
################################## Forecast ###################################
'''
future = 30
initial_result = 700
start_day = 0
initial_date = pd.to_datetime('2020-12-05')

parameters = [paras['Action 13'], paras['Action 3']+paras['Action 7'], paras['Action 11'], paras['Action 12']]
labels = ['Do Nothing','Easing I', 'Tier 2', 'Tier 3']

plt.figure(2, figsize=(20,15))
plt.axvline(x=initial_date, linestyle=':',
            label='Start', linewidth=5, color='r')

for j in range(len(parameters)):
    fit = np.poly1d(parameters[j])

    days = np.array(range(start_day, start_day+future))
    cases = [initial_result]
    future_dates = [initial_date]
    for i in days:
        # value = fit(i)*i + initial_result
        value = np.exp(fit(i)*i)*initial_result
        cases.append(value if value >=0.1 else 0)
        future_dates.append(initial_date + pd.Timedelta(days=i))
        
    plt.plot(future_dates, cases, linewidth=5, label=labels[j])

plt.title('Projection: new cases the next %s days after easing lockdown II (GM)' %future, fontsize=24)
plt.rc('font', size=24)
plt.xlabel('Date', fontsize=24)
plt.ylabel('Cases', fontsize=24)
# plt.ylim([0,3000])
plt.grid(True)
plt.legend(fontsize=24)

'''
############################# multi-scenes ###################################
'''
future = {0: 'Current trend', 
          2: 'Current Lockdown',
          15: 'Tougher Lockdown like I',
          30: 'End of Prediction'}


parameters = [paras['Action 11'], paras['Action 10'],
              paras['Action 2']]

colors = ['black', 'blue', 'green']

cases = [1778]
future_dates = [pd.to_datetime('2021-01-11')]

plt.figure(2, figsize=(20,15))      

for j in range(len(parameters)):
    
    key = list(future.keys())[j]
    
    fit = np.poly1d(parameters[j])

    days = list(range(list(future.keys())[j+1]-key))
    
    start_cases = cases[-1]
    start_date = future_dates[-1]
    
    plt.axvline(x=start_date, linestyle=':',
                label=future[key], linewidth=5, color=colors[j])
    
    for i in days:
        # value = fit(i)*i + initial_result
        value = np.exp(fit(i)*i)*start_cases 
        cases.append(value if value >=0.1 else 0)
        future_dates.append(start_date + pd.Timedelta(days=i))

plt.plot(future_dates, cases, linewidth=5, color='r')
plt.title('Projection: new admissions the next 30 days (with tougher lockdown)', fontsize=24)
plt.rc('font', size=24)
plt.xlabel('Date', fontsize=24)
plt.ylabel('Cases', fontsize=24)
plt.ylim([0,2500])
plt.grid(True)
plt.legend(fontsize=24)












