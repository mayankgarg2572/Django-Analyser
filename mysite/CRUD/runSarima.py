from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import statsmodels.api as sm
from itertools import product

import statsmodels.api as sm
from itertools import product

import warnings
warnings.filterwarnings('ignore')

def return_graph():

    x = np.arange(0,np.pi*3,.1)
    y = np.sin(x)

    fig = plt.figure()
    plt.plot(x,y)

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

def runSarimafff(filenm):
  data = pd.read_csv('jj.csv', parse_dates=['date'])
  data.head()
  plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
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
