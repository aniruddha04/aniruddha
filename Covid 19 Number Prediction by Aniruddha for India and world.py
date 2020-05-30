#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Case Study: Analysing the Outbreak of COVID 19 using Machine Learning
#Problem Statement
#We need a strong model that predicts how the virus could spread across different countries and regions. 
#The goal of this task is to build a model that predicts the spread of the virus till 10th of June

#NOTE: The model was built on a test dataset updated till May 25th. But you can access the 
#source to these datasets at the ‘John Hopkins University Coronavirus Resource Centre’ which gets updated on a daily basis, so you can run this model for the date you prefer.

#Tasks to be performed:
#Analysing the present condition in India
#Exploring the world wide data
#Forecasting the worldwide COVID-19 cases using Prophet for world and India

#importinglibraries

import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins

#plotsize manipulation
plt.rcParams['figure.figsize']=10,12

#disablewarnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Reading the Datasets

India_coord=pd.read_excel('Indian Coordinates.xlsx')
df=pd.read_excel('state_wise1.xlsx')
df_india= df.copy()
df


# In[4]:


#Analysing COVID19 Cases in India

df=pd.read_excel('state_wise1.xlsx')
df_india= df.copy()
df['Total cases']=df['Confirmed']
total_cases= df['Total cases'].sum()
print('Total number of covid confirmed case till 25th may in India:',total_cases)


# In[5]:


#Number of Active COVID-19 cases in affected State/Union Territories

df=pd.read_excel('state_wise1.xlsx')
df_india= df.copy()
df
df.style.background_gradient(cmap='Reds')


# In[6]:


#Visualising the spread geographically

df_full = pd.merge(India_coord,df,on='Name of State / UT')
map = folium.Map(location=[20, 70], zoom_start=4,tiles='Stamenterrain')
for lat, lon, value, name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Confirmed'], df_full['Name of State / UT']):
     folium.CircleMarker([lat, lon], radius=value*0.003, popup = ('<strong>State</strong>: ' + str(name).capitalize() + ''),color='red',fill_color='red',fill_opacity=0.3 ).add_to(map)

map


# In[7]:


#Confirmed vs Recovered figures

f, ax = plt.subplots(figsize=(12,8))
data= df_full[['Name of State / UT','Confirmed','Recovered','Deaths']]
data.sort_values('Confirmed',ascending=False,inplace=True)
sns.set_color_codes("pastel")
sns.barplot(x="Confirmed",y="Name of State / UT",data=data,label="Total",color="r")

sns.set_color_codes("muted")
sns.barplot(x="Recovered",y="Name of State / UT",data=data,label="Cured",color="g")

ax.legend(ncol=2, loc= "lower right", frameon=True)
ax.set(xlim=(0,35000), ylabel="", xlabel="Cases")
sns.despine(left=True,bottom=True)


# In[8]:


#Exploring Worldwide Data

df = pd.read_csv('covid_19_clean_complete.csv',parse_dates=['Date'])
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)
df_confirmed = pd.read_csv("time_series_covid19_confirmed_global.csv")
df_recovered = pd.read_csv("time_series_covid19_recovered_global.csv")
df_deaths = pd.read_csv("time_series_covid19_deaths_global.csv")
df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.head()


# In[9]:


df2 = df.groupby(["Date", "Country", "Province/State"])[['Date', 'Province/State', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df2.head()


# In[10]:


df.query('Country=="India"').groupby("Date")[['Confirmed','Deaths','Recovered']].sum().reset_index()


# In[11]:


df.groupby("Date").sum().head()


# In[12]:


confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()


# In[13]:


#Forecasting Total Number of Cases Worldwide
#In this segment, we’re going to generate a week ahead forecast of 
#confirmed cases of COVID-19 using Prophet, with specific prediction 
#intervals by creating a base model both with and without tweaking of 
#seasonality-related parameters and additional regressors.

from fbprophet import Prophet


# In[14]:


confirmed=df.groupby('Date').sum()['Confirmed'].reset_index()
deaths=df.groupby('Date').sum()['Deaths'].reset_index()
recovered=df.groupby('Date').sum()['Recovered'].reset_index()


# In[15]:


confirmed.columns=['ds','y']
confirmed['ds']=pd.to_datetime(confirmed['ds'])


# In[16]:


confirmed.tail()


# In[17]:


#Forecasting Confirmed COVID-19 Cases Worldwide with Prophet (Base model)

m = Prophet(interval_width=0.95) 
m.fit(confirmed) 
future = m.make_future_dataframe(periods=16) 
future.tail()


# In[18]:


#predicting the future with date, and upper and lower limit of y value

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[19]:


confirmed_forecast_plot=m.plot(forecast)


# In[20]:


confirmed_forecast_plot =m.plot_components(forecast)


# In[21]:


#Forecasting Worldwide Recovered using Prophet (Base model)

recovered.columns=['ds','y']
recovered['ds']=pd.to_datetime(confirmed['ds'])


# In[22]:


recovered.tail()


# In[23]:


m = Prophet(interval_width=0.95) 
m.fit(recovered) 
futurerecovered = m.make_future_dataframe(periods=16) 
futurerecovered.tail()


# In[24]:


##predicting the future with date, and upper and lower limit of y value

forecastrecovered = m.predict(future)
forecastrecovered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[25]:


confirmed_forcast_plot=m.plot(forecastrecovered)


# In[26]:


confirmed_forecastrecovered_plot =m.plot_components(forecastrecovered)


# In[27]:


#Forecasting Worldwide Deaths using Prophet (Base model)

deaths.columns=['ds','y']
deaths['ds']=pd.to_datetime(confirmed['ds'])


# In[28]:


deaths.tail()


# In[29]:


m = Prophet(interval_width=0.95) 
m.fit(deaths) 
futuredeaths = m.make_future_dataframe(periods=16) 
futuredeaths.tail()


# In[30]:


#predicting the future with date, and upper and lower limit of y value

forecastdeaths = m.predict(future)
forecastdeaths[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[31]:


death_forcast_plot=m.plot(forecastdeaths)


# In[32]:


confirmed_forecastdeath_plot =m.plot_components(forecastdeaths)


# In[75]:


dfdateso=forecast['ds']
dfconfirmedo=forecast['yhat']
dfrecoveredo=forecastrecovered['yhat']
dfdeathso=forecastdeaths['yhat']


# In[76]:


import numpy as np
plt.plot(dfdateso, dfconfirmedo,label='Confirmed')
plt.plot(dfdateso, dfrecoveredo,label='recovered')
plt.plot(dfdateso, dfdeathso,label='deaths')
plt.legend()


# In[33]:


#Forecasting Indian Corona numbers using Prophet model

df2=df.query('Country=="India"').groupby("Date")[['Confirmed','Deaths','Recovered']].sum().reset_index()


# In[34]:


df2.groupby('Date').sum().tail()


# In[35]:


confirmedindia = df2.groupby('Date').sum()['Confirmed'].reset_index()
deathsindia = df2.groupby('Date').sum()['Deaths'].reset_index()
recoveredindia = df2.groupby('Date').sum()['Recovered'].reset_index()


# In[36]:


from fbprophet import Prophet


# In[37]:


confirmedindia = df2.groupby('Date').sum()['Confirmed'].reset_index()
deathsindia = df2.groupby('Date').sum()['Deaths'].reset_index()
recoveredindia = df2.groupby('Date').sum()['Recovered'].reset_index()


# In[38]:


#Forecasting Confirmed COVID-19 Cases in India with Prophet (Base model)

confirmedindia.columns=['ds','y']
confirmedindia['ds']=pd.to_datetime(confirmedindia['ds'])


# In[39]:


confirmedindia.tail()


# In[40]:


m = Prophet(interval_width=0.95) 
m.fit(confirmedindia) 
future = m.make_future_dataframe(periods=16) 
future.tail()


# In[45]:


##predicting the future with date, and upper and lower limit of y value

forecastconfirmedindia = m.predict(future)
forecastconfirmedindia[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[44]:


confirmedindia_forcast_plot=m.plot(forecastconfirmedindia)


# In[47]:


confirmed_forecastindia_plot =m.plot_components(forecastconfirmedindia)


# In[48]:


#Forecasting India Recovered Cases with Prophet (Base model)

recoveredindia.columns=['ds','y']
recoveredindia['ds']=pd.to_datetime(recoveredindia['ds'])


# In[49]:


recoveredindia.tail()


# In[50]:


m = Prophet(interval_width=0.95) 
m.fit(recoveredindia) 
future = m.make_future_dataframe(periods=16) 
future.tail()


# In[51]:


##predicting the future with date, and upper and lower limit of y value

forecastrecoveredindia = m.predict(future)
forecastrecoveredindia[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[52]:


recoveredindia_forcast_plot=m.plot(forecastrecoveredindia)


# In[53]:


recovered_forecastindia_plot =m.plot_components(forecastrecoveredindia)


# In[54]:


#Forecasting Deaths in India using Prophet (Base model)

deathsindia.columns=['ds','y']
deathsindia['ds']=pd.to_datetime(deathsindia['ds'])


# In[55]:


deathsindia.tail()


# In[56]:


m = Prophet(interval_width=0.95) 
m.fit(deathsindia) 
future = m.make_future_dataframe(periods=16) 
future.tail()


# In[57]:


#predicting the future with date, and upper and lower limit of y value

forecastdeathsindia = m.predict(future)
forecastdeathsindia[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[58]:


deathsindia_forcast_plot=m.plot(forecastdeathsindia)


# In[59]:


deaths_forecastindia_plot =m.plot_components(forecastdeathsindia)


# In[68]:


dfdates=forecastconfirmedindia['ds']
dfconfirmedindia1=forecastconfirmedindia['yhat']
dfrecoveredindia1=forecastrecoveredindia['yhat']
dfdeathsindia1=forecastdeathsindia['yhat']


# In[73]:


import numpy as np
plt.plot(dfdates, dfconfirmedindia1,label='Confirmed')
plt.plot(dfdates, dfrecoveredindia1,label='recovered')
plt.plot(dfdates, dfdeathsindia1,label='deaths')
plt.legend()


# In[ ]:


#Conclusion
#This is a humble request to all our learners.
#Don’t take your cough and cold lightly as you would. If you look at the data, the number of cases in India is rising just like in USA. We will reach  mark of 200,000 cases by 10th June. Don’t let lower awareness and fewer test numbers ruin the health of our world.
#But the Best part here is Recovery rate is rising at the better pace as compared to confirmed cases it can reach mark of 82000 by 10th of June.
#It shows us that if there are 100 patients getting admitted on 1st day on the 14th day around 97 patients will receive discharge ie.2.8% fatality rate
#This Data shows there will be a time in net few months when government will declare covid-19 as a normal flu in comparison to no. of recoveries and give permissions to business to re-open
#New rules of living will be made where they will give us list of precautions we should take while we interact with people outside our homes.
#Let’s give a hand in fighting this pandemic at least by quarantining ourselves by staying indoors and protecting ourselves and others around us.
#Take precautions and stay indoors.

