from itertools import product
import statsmodels.api as sm
from io import StringIO
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm 
from tqdm import tqdm

from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
from django.shortcuts import render
from polls.models import Sale
from django.shortcuts import redirect
from csvs.models import Csv
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.decorators import login_required
import csv

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import warnings
warnings.filterwarnings('ignore')
# from runSarima import runSarimafff
# Create your views here.
 
def error_404_view(request, exception):
   
    # we add the path to the 404.html file
    # here. The name of our HTML file is 404.html
    return render(request, '404.html')
# @login_required(redirect("accounts:login"))
@login_required(login_url="/accounts/login/")
def INDEX(request):
    sale = Sale.objects.all()
    context = {
        'sale': sale,
    }
    return render(request, 'CRUD/index.html', context)




@login_required(login_url="/accounts/login/")
def ADD(request):
    if (request.method == 'POST'):
        print(request)
        prod_desc = request.POST.get('prod_desc')
        cost = request.POST.get('cost')
        date_of_pur = request.POST.get('date_of_pur')

        sale = Sale.objects.create(
            prod_desc=prod_desc,
            cost=cost,
            date_of_pur=date_of_pur
        )

        sale.save()
        return redirect('CRUD:home')
    sale = Sale.objects.all()
    context = {
        'sale': sale,
    }
    return render(request, 'CRUD/index.html', context)


@login_required(login_url="/accounts/login/")
def EDIT(request):

    sale = Sale.objects.all()
    print(type(sale[0].date_of_pur))
    context = {
        'sale': sale,
    }
    return render(request, 'CRUD/index.html', context)


@login_required(login_url="/accounts/login/")
def UPDATE(request, id):
    if (request.method == 'POST'):
        prod_desc = request.POST.get('prod_desc')
        cost = request.POST.get('cost')
        date_of_pur = request.POST.get('date_of_pur')
        sale = Sale(
            id=id,
            prod_desc=prod_desc,
            cost=cost,
            date_of_pur=date_of_pur
        )
        sale.save()
        return redirect('CRUD:home')
    sale = Sale.objects.all()
    context = {
        'sale': sale,
    }
    return render(request, 'CRUD/index.html', context)


@login_required(login_url="/accounts/login/")
def DELETE(request, id):
    sale = Sale.objects.filter(id=id).delete()
    return redirect('CRUD:home')


@login_required(login_url="/accounts/login/")
def UPDATECSV(request, id):
    csv = Csv.objects.filter(id=id).update(activated=True)
    return redirect('CRUD:csvfiles')


@login_required(login_url="/accounts/login/")
def DELETECSV(request, id):
    csv = Csv.objects.filter(id=id).delete()
    return redirect('CRUD:csvfiles')

@login_required(login_url="/accounts/login/")
def showaCsv(request, id):
    print(id)
    tmpcsv = Csv.objects.filter(id=id).first()
    print(tmpcsv)
    filename = tmpcsv.file_name.path
    print(filename)
    filedata = []
    header = []
    with open(filename, 'r') as f:
        print(f)
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                header = row

            else:
                filedata.append(row)
    context = {
        "filedata": filedata,
        "header": header,
        "id": id
    }
    return render(request, 'CRUD/showacsv.html', context)


@login_required(login_url="/accounts/login/")
def showallCSV(request):
    csvs = Csv.objects.all()
    filesdata = []
    for obj in csvs.iterator():
        tmpobj = {
            "id": obj.id,
            "filename": obj.file_name,
            "fileUploadDateTime": obj.uploaded,
            "activated": obj.activated,
            "path": obj.file_name.path,
        }
        filesdata.append(tmpobj)
    context = {
        "filesData": filesdata
    }
    return render(request, 'CRUD/showallCSV.html', context)

# From here our main work will start that is analysing a CSV file using the time series algorithm
# We will use SARIMA algorithm for this purpose

def Basic_dataGraph(data):
    fig = plt.figure(figsize=[8, 6])  # Set dimensions for figure
    plt.plot(data['date'], data['data'])
    plt.title('Quarterly EPS for Johnson & Johnson')
    plt.ylabel('EPS per share ($)')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.grid(True)
    imgdata = StringIO()

    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    return data


def acf_plot(data):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    sm.graphics.tsa.plot_pacf(data['data'].values.squeeze(), lags=10, ax=ax[0])
    sm.graphics.tsa.plot_acf(data['data'].values.squeeze(), lags=10, ax=ax[1])
    plt.plot()
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data2 = imgdata.getvalue()
    return data2

def kpss_test(timeseries):
    result = {}
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
      kpss_output['Critical Value (%s)'%key] = value
    return kpss_output

# de fresults before the actual analysis
def log_dif_plot(data, dif =1):
    data['log_data'] = np.log(data['data'])
    data['log_dif_data'] = data['log_data'].diff(int(dif))
    data = data.iloc[int(dif):]
    fig = plt.figure(figsize=[8, 6]); # Set dimensions for figure
    plt.plot(data['log_dif_data'])
    plt.title("Log Difference of Quarterly EPS for Johnson & Johnson")
    # plt.plot()
    imgdata = StringIO()

    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data1 = imgdata.getvalue()

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    sm.graphics.tsa.plot_pacf(data['log_dif_data'].values.squeeze(), lags=10, ax=ax[0])
    sm.graphics.tsa.plot_acf(data['log_dif_data'].values.squeeze(), lags=10, ax=ax[1])
    plt.plot()
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data2 = imgdata.getvalue()
    logDif = {}
    logDif['acf'] = data2
    logDif['basic'] = data1
    return logDif


def basic_results(data):
    # Here we will do the data visualizatikon of the data preanalysis\
    # For that here we will be doing the first the time respective plot of the data 
    # Then we will do the ACF and PACF plot of the data
    # Then we will do the KPSS and the adFuller test of the data
    # and the differentiating the data and then again doing the ACF and PACF plot of the data
    # and then again doing the KPSS and the adFuller test of the data
    
    plots = {}
    plots['basicPlot'] =Basic_dataGraph(data)
    plots['acf'] = acf_plot(data)

    plots['kpssresuult'] = kpss_test(data['data'])
    plots['adfullerresult'] = adfuller(data['data'])
    plots['log_dif_plot'] = log_dif_plot(data)

    return plots
    # then we will provide an option for user to select the differencing the data upto a certain limit and then we will show the user the graph for the user accordingly

def runSarimafff(path):
    data = pd.read_csv(path, parse_dates=['date'])
    data.head()
    basicresult = basic_results(data)
    return basicresult
    # The data visualization before the analysis is done since w eneed to get the value or the range of 
    # p q r s and P Q R S from the user after user sees the graphs and then we will run the algorithm
    # on the csv data and provide the analysis using the SARIMA algorithm


    # plots['pacf'] = pacf_plot(data)
    # return plots



@login_required(login_url="/accounts/login/")
def runSarimaView(request, id):
    print(id)
    filePath = Csv.objects.filter(id=id).first().file_name.path
    context = {}
    context['dif'] = 1
    context["id"] = id
    context['graph'] = runSarimafff(filePath)
    return render(request, 'CRUD/showGraph.html', context)


@login_required(login_url="/accounts/login/")
def varDifPlot(request, id):
    print(id)
    filePath = Csv.objects.filter(id=id).first().file_name.path
    dif = int(request.POST.get('dif'))
    context = {}
    context["id"] = id
    context['dif'] = dif
    data = pd.read_csv(filePath, parse_dates=['date'])
    plots = {}
    plots['basicPlot'] =Basic_dataGraph(data)
    plots['acf'] = acf_plot(data)

    plots['kpssresuult'] = kpss_test(data['data'])
    plots['adfullerresult'] = adfuller(data['data'])
    plots['log_dif_plot'] = log_dif_plot(data, dif)
    context['graph'] = plots
    return render(request, 'CRUD/showGraph.html', context)







def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """
    
    results = []
    best_param= []
    best_aic = 10100101
    for param in tqdm(parameters_list):
        try: 
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        if(aic < best_aic):
            best_aic = aic
            best_param = [param[0], d, param[1], param[2], D, param[3], s]
        results.append([param, aic])
    print(results)    
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df, best_param

def parameterListCreater(prange, qRange, dif, season):
    
    p = (1) if len(prange)==1 else range(prange[0],  prange[1], 1)
    d = dif
    q = (1) if len(qRange)==1 else range(qRange[0],qRange[1], 1)
    print(p, q)
    P = p
    D = d
    Q = q
    s = season
    parameters = product(p, q, P, Q)
    print("parameters:", parameters)
    parameters_list = list(parameters)
    print("parameters list:", parameters_list)
    return parameters_list

def forecastGraph(data, bestModel):
    print(data)
    # date_format = "%Y-%d-%m"  
    # last_date = pd.to_datetime(data['date'].iloc[-1], format=date_format)
    # second_last_date = pd.to_datetime(data['date'].iloc[-2], format=date_format)
    
    modelVal =  bestModel.fittedvalues
    print(modelVal)
    data['arima_model'] = modelVal
    # data['arima_model'][:4+1] = np.NaN
   # Calculate the start date for forecasting
    # last_date = data['date'].iloc[-1]
    next_month = data['date'].iloc[-1]
    
    # Forecast 9 steps ahead with 3-month intervals
    # date_format = "%Y-%d-%m"  
    # last_date = pd.to_datetime(data['date'].iloc[-1], format=date_format)
    # second_last_date = pd.to_datetime(data['date'].iloc[-2], format=date_format)
    date_format = "%Y/%d/%m"  # Change this to your date format
    last_date = pd.to_datetime(data['date'].iloc[-1], format=date_format)
    second_last_date = pd.to_datetime(data['date'].iloc[-2], format=date_format)
    print(type(data['date'].iloc[-1]), type(data['date'].iloc[-2]))
    date_gap = (  second_last_date - last_date).days
    res = (dt.strptime(data['date'].iloc[-2].strftime("%Y/%d/%m"), "%Y/%d/%m") - dt.strptime(data['date'].iloc[-1].strftime("%Y/%d/%m"), "%Y/%d/%m")).days
    print(last_date, second_last_date, date_gap, res)
    print(type(last_date), type(second_last_date), type(date_gap), type(res))
    # Generate forecasted dates
    forecast_start_date = last_date + pd.DateOffset(days=1)
    forecast_end_date = forecast_start_date + pd.DateOffset(days=9 * date_gap)  # Forecast 9 steps ahead
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq=f'{date_gap}D')
    # forecast_index = pd.date_range(start=next_month, periods=forecast_steps, freq=f'{date_gap}D')
    forecast = bestModel.get_forecast(steps=len(forecast_dates))
    
    forecast_df = pd.DataFrame({'date': forecast_dates, 'predicted_mean': forecast.predicted_mean.values})
    forecast_df.set_index('date', inplace=True)
    
    forecast = pd.concat([data.set_index('date'), forecast_df], axis=1)
    print(forecast)
    # forecast = pd.concat([data, forecast], axis=1)
    # forecast = data['arima_model'].append(forecast)
    
    fig =  plt.figure(figsize=(8, 6))
    plt.plot(forecast['predicted_mean'], label='prediction')
    plt.plot(forecast['arima_model'], label='sarima_model')
    plt.plot(forecast['data'],  label='actual')
    # plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    # plt.plot(data['data'], label='actual')
    plt.legend()
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data2 = imgdata.getvalue()
    # forecast.index = forecast.index.strftime('%Y-%d-%m')
    print(type(forecast))
    csv_data = forecast.to_csv()
    return data2, csv_data

def dianoSticGraph(best_Model):
    fig = best_Model.plot_diagnostics(figsize=(15,12))
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data2 = imgdata.getvalue()
    return data2

def runSarimaAnalysis(req, id):
    context = {}
    # we will be getting the P and Q ranges based on the graph plot from the user in this function we willl also we getting is log necesarry for the data
    # and if it  is then how much dif of log is required that we will be getting  that is 
    # d
    # S
    # pRange Start and end 
    # qRange Start and end
    # dif
    # and the id of the CSV file that I will be 
    pStart, pEnd, qStart, qEnd, dif, ses =  int(req.POST.get('pStart')), int(req.POST.get('pEnd')), int(req.POST.get('qStart')), int( req.POST.get('qEnd')), int(req.POST.get('diff')), int(req.POST.get('season'))
    # dif = 4
    filePath = Csv.objects.filter(id=id).first().file_name.path
    # "D:\SelfProject\Nitesh_Sr_project\Django-Analyser\Django-Analyser/media/csvs"+
    data = pd.read_csv(filePath, parse_dates=['date'])
    # data['log_data'] = 
    # data['data'] =  np.log(data['data'])bghugf[;.]
    # data['data'] = data['data'].diff(dif)
    data = data.drop(list(range(1, dif, 1)), axis=0).reset_index(drop=True)
    pL = parameterListCreater([int(pStart),int(pEnd)], [ int(qStart), int(qEnd)], int(dif), int(ses))
    print("parameter List:", pL)
    result_df, best_prm = optimize_SARIMA(pL, qStart, dif, ses, data['data'])
    context["result_df"] = result_df
    # from result dataframe we need to use the best parameter using theri AIC values
    print(result_df)
    data = data.dropna()
    best_model = SARIMAX(data['data'], trend =  'c', order=(best_prm[0], best_prm[1], best_prm[2]), seasonal_order=(best_prm[3], best_prm[4], best_prm[5], best_prm[6]), enforce_stationarity=False, enforce_invertibility=False).fit(dis=-1)
    context['modelSummary'] = best_model.summary
    print(context['modelSummary'])
    context["ForecastGraph"], csv = forecastGraph(data, best_model)
    context['csv_data'] = csv
    context['diagnostic'] = dianoSticGraph(best_model)
    return render(req, 'CRUD/shoeSarimaAnalysis.html', context)





'''
So what I will do is that:
I will make HTML file where all the basic details of the database will be shown like a log variance graph and a custom log variance graph which will give the result as per the user selected difference and will throw the output in that difference only The output will consist of three graphs 

first graph will be the simple log dif graph of the data, second and third graph will be the pacf and acf plot from there user will get the option to fix the

ranges of the p q r d and P Q R D values and hence we will then show the main output in a new HTML file 
The new HTML file will consist of diagnostic plot and 


'''

'''
To activate the venv:  my_project_venv\Scripts\activate


'''