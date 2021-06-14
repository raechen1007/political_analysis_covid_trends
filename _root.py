#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:44:33 2020

@author: raechen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:47:05 2020

@author: raechen
"""
from requests import get
import pandas as pd
from numpy import nan
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from scipy.optimize import curve_fit

'''
############################Load Functions#####################################
'''

#Extract data from goverment API
def _getAPIData(endpoint):
    response = get(endpoint)
    
    if response.status_code >= 400:
        raise RuntimeError(f'Request failed: { response.text }')
        
    return response.json()

def GetLTLAData(district):
    endpoint = (
            'https://api.coronavirus.data.gov.uk/v1/data?'
            'filters=areaType=ltla; areaName='+district+'&'
            'structure={"Date":"date","NewCases":"newCasesBySpecimenDate"'
            ',"TotalCases":"cumCasesBySpecimenDate", "District": "areaName"}'
            )
        
    data = _getAPIData(endpoint)
    data = pd.DataFrame(data['data'])
    
    data.Date = pd.to_datetime(data.Date, format='%Y-%m-%d')
    data = data.sort_values(by='Date')
    data = data[data.Date <= (data.Date.max() - pd.Timedelta('3 days'))]
    
    return data


def GetRegionData(region):
    endpoint = (
                'https://api.coronavirus.data.gov.uk/v1/data?'
                'filters=areaType=region; areaName='+region+'&'
                'structure={"Date":"date","NewCases":"newCasesByPublishDate"'
                ',"TotalCases":"cumCasesBySpecimenDate", "District": "areaName"}'
                )
        
    data = _getAPIData(endpoint)
    data = pd.DataFrame(data['data'])
    
    data.Date = pd.to_datetime(data.Date, format='%Y-%m-%d')
    data = data.sort_values(by='Date')
    data = data[data.Date <= (data.Date.max() - pd.Timedelta('3 days'))]
    
    return data
    
def GetGMData():
    '''

    Returns
    -------
    data : DataFrame
        number of cases/total cases of 10 districts in GM

    '''
    dist_list = ("Manchester", "Trafford", "Bury", "Tameside", "Rochdale", 
                 "Salford", "Stockport", "Wigan", "Bolton", "Oldham")
    data_list = []
    for district in dist_list:
        endpoint = (
            'https://api.coronavirus.data.gov.uk/v1/data?'
            'filters=areaType=ltla; areaName='+district+'&'
            'structure={"Date":"date","NewCases":"newCasesBySpecimenDate"'
            ',"TotalCases":"cumCasesBySpecimenDate", "District": "areaName"}'
        )
        
        data = _getAPIData(endpoint)
        data = data['data']
        data_list.append(pd.DataFrame(data))
            
    data = pd.concat(data_list, ignore_index=True)
    data.Date = pd.to_datetime(data.Date, format='%Y-%m-%d')
    
    GMdata = data
    
    for time in data.Date.unique():
        GMdata = GMdata.append({'Date': time, 
                                'NewCases': sum(data[data.Date == time].NewCases),
                                'TotalCases': sum(data[data.Date == time].TotalCases),
                                'District': 'GM'}, ignore_index=True)
        
    GMdata = GMdata.sort_values(by='Date')
    
    GMdata = GMdata[GMdata.Date <= (GMdata.Date.max() - pd.Timedelta('3 days'))]
    
    return GMdata

def reformFactorisedlTimeline(district, factorised_timeline, factors):
    local_df = factorised_timeline[factorised_timeline['District'] == district].reset_index(drop=True)
    factor_boo = np.array(local_df[factors])
    
    local_timeline = pd.DataFrame(columns=(['Date', 'Interventions', district, 'Description']))
    
    i,j = 1, 1
    actions_fct = {'Action 1': factor_boo[0]} #start from lockdown
    
          
    txt = []         
    for k in range(len(factor_boo[0])):    
        if factor_boo[0][k]:
            txt.append(factors[k])
    
    
    local_timeline = local_timeline.append({'Date':local_df['Date'][0],
                                   'Interventions': 'Action 1',
                                   district: True,
                                   'Description': ' + '.join(txt)},
                                   ignore_index=True)
    
    while i < factor_boo.shape[0]:
        if any(factor_boo[i] != factor_boo[i-1]):
            j += 1
            actions_fct['Action '+str(j)] = factor_boo[i]
            
            txt = []         
            for k in range(len(factor_boo[i])):    
                if factor_boo[i][k]:
                    txt.append(factors[k])
                    
            local_timeline = local_timeline.append({'Date':local_df['Date'][i],
                                            'Interventions': 'Action '+str(j),
                                            district: True,
                                            'Description': ' + '.join(txt)},
                                           ignore_index=True)
            
        
        i += 1
    
    local_timeline = local_timeline.set_index('Date')
    return actions_fct, local_timeline

def TimelineShift(timeline, shift):
    
    shift_indices = [pd.to_datetime(date)+pd.Timedelta(days=shift) for date in timeline.index]
    timeline = timeline.set_index(pd.Index(shift_indices))
    
    
    return timeline

def plotInterventions(data, timeline, by, district, shift=0):
    '''

    Parameters
    ----------
    data : DataFrame 
        data from GetDFByDist
    timeline : DataFrame
        intervention timeline table from file
    district : string
        Name of district
    by : string
        NewCases or TotalCases

    Returns
    -------
    plots

    '''
    
    local_data = data[data["District"] == district]
    local_data = local_data.sort_values(by='Date')
    local_data = local_data.set_index('Date')    
    
    plt.figure(1, figsize=(30,20))
    plt.plot(local_data[by], label=by, linewidth=6)
    
    timeline = TimelineShift(timeline, shift)
    timeline = timeline['Interventions'][timeline[district]==True]

    checker = -1
    while timeline.index[checker] >= (data['Date'].max() - pd.Timedelta(days=2)):
        timeline = timeline.drop(timeline.index[checker])
        checker -= 1    
    
    for i in range(len(timeline)):
        plt.scatter(timeline.index[i], local_data[by][timeline.index[i]])
        plt.annotate(timeline[timeline.index[i]], 
                     (timeline.index[i], local_data[by][timeline.index[i]]),
                     fontsize=24)

    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Change of ' + by + ' in ' + district, fontsize=24)
    plt.rc('font', size=20)
    plt.grid(True)
    plt.rc('font', size=26)
    plt.show()
    
    return local_data
    
def AppendVariables(data, timeline, district, scenario=1, shift=0):
    data = data[data.District == district].sort_values(by='Date', ignore_index=True)
    
    #append days
    data['Days'], i = 0, 1
    while i < len(data):
        data.loc[i, 'Days'] = i + (data['Date'][i] - data['Date'][i-1]).days - 1
        i += 1
    
    timeline = TimelineShift(timeline, shift)
    local_timeline = timeline['Interventions'][timeline[district]==True]
    
    interventions = list(local_timeline.unique())
    
    data = pd.concat([data, pd.DataFrame(columns = interventions)])
    data = data.replace(nan, 0)
    
    #Scenario1 (default): consider the impact of interventions last always
    for i in range(len(interventions)):
        date = local_timeline.index[pd.Index(local_timeline).get_loc(interventions[i])]
        data[interventions[i]][data['Date'] >= date] = 1
     
    if scenario == 2:
    #Scenario2: consider the impact of interventions last until next intervention
        for i in range(1, len(interventions)):
            date = local_timeline.index[pd.Index(local_timeline).get_loc(interventions[i])]
            data.loc[data['Date'] >= date, interventions[i - 1]] = 0
            
    return data

def PolySmooth(data, smooth):
    
    data = data.copy()
    
    x, y = data['Days'], data['NewCases']
    pln = np.polyfit(x, y, smooth)
    pln_y = np.poly1d(pln)(x).astype(int)
    pln_y[pln_y < 0] = 0  
    data['NewCases'] = pln_y
    
    plt.figure(0, figsize=(10,6))
    plt.plot(data['Days'], data['NewCases'])
    plt.plot(data['Days'], pln_y)
    
    
    return data
         

def InterventionFitting(local_data, timeline, method, by, district, shift=0, 
                        smooth=0, degree=1, last=15, reset=False, tolerance=20,
                        plot=True, add_description=True):
    
    data = local_data.copy()
    
    if smooth > 0: #use polyfit remove spikes in NewCases
        x, y = data['Days'], data['NewCases']
        pln = np.polyfit(x, y, smooth)
        pln_y = np.poly1d(pln)(x).astype(int)
        pln_y[pln_y < 0] = 0  
        data['NewCases'] = pln_y
        
    fit_y = []
    predict_x = []
    predict_y = []
    
    diff = []
    paras = {}    
    
    timeline = TimelineShift(timeline, shift)
    local_timeline = timeline[['Interventions', 'Description']][timeline[district]==True].sort_index()
    
    checker = -1
    while local_timeline.index[checker] >= (data['Date'].max() - pd.Timedelta(days=2)):
        local_timeline = local_timeline.drop(local_timeline.index[checker])
        checker -= 1

    if add_description:
        textstr = []
        for j in range(len(local_timeline)):
            textstr.append(str(local_timeline['Interventions'][j])+' : '+str(local_timeline['Description'][j]))  
        textstr = '\n'.join(textstr)
    
    date_cut = list(local_timeline.index)
    date_cut.insert(0 ,list(data.Date)[0])
    date_cut.append(list(data.Date)[-1])
    
    labels = list(local_timeline['Interventions'].values)
    labels.insert(0, 'Start')
    
    for i in range(len(date_cut)-1):
        start = date_cut[i]
        end = date_cut[i+1]
        interved_df  = data[(data.Date > start)&(data.Date <= end)][['Date', by, 'Days']]
        
        if reset: #suppose the number of days has no impact
            initial_day = interved_df['Days'].min()    
            initial_result = interved_df[by][interved_df['Days'] == initial_day].values[0] #the first-day result
            interved_df['Days'] = interved_df['Days'] - initial_day
            
            if not method=='exp':
                interved_df[by] = interved_df[by]-initial_result
            
        x = np.array(interved_df['Days']).astype(int)
        y = np.array(interved_df[by]).astype(int)
        
        interval = np.append(x, np.array([i for i in range(x[-1], x[-1]+last+1)]))
        
        if method == 'polynomial':
            
            if reset:
                parameters = np.polyfit(x[1:], y[1:]/x[1:], degree-1)
                
                fit_y.append(np.poly1d(parameters)(x)*x + initial_result)
                predict_y.append(np.poly1d(parameters)(interval)*interval + initial_result)
                
                parameters = np.append(parameters, 0)
                
            else:
                parameters = np.polyfit(x, y, degree)
                
                fit_y.append(np.poly1d(parameters)(x))
                predict_y.append(np.poly1d(parameters)(interval)*interval)         

        # if method == 'log':
        #     parameters =  np.polyfit(np.log(x), y, degree)
            
        #     if reset:
        #         fit_y.append(np.poly1d(parameters)(np.log(x)) + initial_result) 
        #         predict_y.append((np.poly1d(parameters)(np.log(interval)) + initial_result))
                
        #     else:    
        #         fit_y.append(np.poly1d(parameters)(np.log(x)))
        #         predict_y.append(np.poly1d(parameters)(np.log(interval)))
            
        if method == 'exp':
            
            def exp_fun(x, a):
                return np.exp(a * x)
   
            if reset:
                y = np.where(y == 0, 0.1, y)
                if initial_result==0:
                    initial_result = 0.1
                
                z = y[1:]/initial_result
                
                parameters, _ = curve_fit(exp_fun,  x[1:],  z)
                
                fit_y.append(np.exp(parameters[0]*x)*initial_result)
                predict_y.append(np.exp(parameters[0]*interval)*initial_result)
                
                # z = (np.log(y[1:]/initial_result))/x[1:]
                # parameters =  np.polyfit(x[1:], z, degree-1)
                
                # fit_y.append(np.exp(np.poly1d(parameters)(x)*x)*initial_result)
                # predict_y.append(np.exp(np.poly1d(parameters)(interval)*interval)*initial_result)
            
            else:
                y = np.where(y == 0, 0.1, y)
                parameters, _ = curve_fit(exp_fun,  x,  y)
                
                fit_y.append(np.exp(parameters[0]*x))
                predict_y.append(np.exp(parameters[0]*interval))
         
        if reset:
            real_y = data[by][(data['Days'] >= (interval[0] + initial_day)) 
                              & (data['Days'] <= (interval[-1]+ initial_day))].values
        else:
            real_y = data[by][(data['Days'] >= interval[0]) & (data['Days'] <= interval[-1])].values
        
        day = len(x)
        while day < len(interval):
            if (predict_y[-1][day] > tolerance or predict_y[-1][day]<0):
                interval = interval[:day]
                predict_y[-1] = predict_y[-1][:day]
            
            day += 1
  
        predict_x.append([start + pd.Timedelta(days=x) for x in range(len(interval))])
        
        diff.append({'Intervention':labels[i],
                     'Expected': predict_y[-1][-1],
                     'Real': real_y[-1],
                     'Difference': (predict_y[-1][-1]-real_y[-1])/real_y[-1]})
        
        paras[labels[i]] =  parameters
    
    fit_y = np.concatenate(fit_y).ravel()
    diff = pd.DataFrame(diff)
    
    """
    Plotting procedure
    """
    if plot:
        
        plt.figure(1, figsize=(20,10))
        plt.plot(local_data['Date'], local_data[by], linewidth=6, label='Actual', 
                 alpha=.2, color='green')
        if smooth > 0:
            plt.plot(data['Date'], data[by], linewidth=5, label='Curve', 
                      color='purple', alpha=.5)
        plt.plot(data.Date[1:], fit_y, '-', linewidth=3, label='Fit', color='orange')
        
        for j in range(len(local_timeline)):
            plt.scatter(local_timeline.index[j], data[by][data.Date == local_timeline.index[j]])
            plt.annotate(local_timeline['Interventions'][local_timeline.index[j]], 
                         (local_timeline.index[j], data[by][data.Date == local_timeline.index[j]]))
        plt.xlabel('Date', fontsize=22)
        plt.ylabel('Count', fontsize=22)
        plt.title('Actual and Fit of '+by+' in '+district+' using '+method+' method', fontsize=22)
        plt.rc('font', size=22)
        plt.grid(True)
        plt.legend(fontsize=22)
        plt.show()
        
        plt.figure(2, figsize=(30,15))
        plt.plot(local_data['Date'], local_data[by], linewidth=6, label='Actual', 
                 alpha=.2, color='green')
        plt.plot(data['Date'], data[by], linewidth=5, label='Curve')
        for i in range(len(predict_x)):
            plt.plot(predict_x[i], predict_y[i], ':', linewidth=5, label=labels[i])  
        for j in range(len(local_timeline)):
            plt.scatter(local_timeline.index[j], data[by][data.Date == local_timeline.index[j]])
            plt.annotate(local_timeline['Interventions'][local_timeline.index[j]], 
                         (local_timeline.index[j], data[by][data.Date == local_timeline.index[j]]),
                         fontsize=26)
        plt.xlabel('Date', fontsize=30)
        plt.ylabel('Count', fontsize=30)
        if add_description:
            plt.gcf().text(0.13,-0.2, textstr, fontsize=26, 
                           bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
        plt.title('Actual and Predict of '+ by +' in '+district+' using '+method+' method', fontsize=30)
        plt.rc('font', size=26)
        plt.grid(True)
        plt.legend(fontsize=26, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.show()
    
    return diff, paras, data