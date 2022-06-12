from re import S
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import calendar
import streamlit as st
import streamlit_option_menu as stm
import altair as alt
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#---------------------------------------------------------------------------------------------------
## FUNCTIONS

# function returning the dataframe corresponding to the day,month,year,weekday,site
def df_by_day_month_year_site_weekday(day,month,year,weekday,site):
  if day!='Not Specified' :
    if site!='Not Specified':
      df = dataframes[year][(dataframes[year]['visit_date'].dt.day == day) & (dataframes[year]['visit_date'].dt.month == month) & (dataframes[year]['site'] == site)].reset_index().drop(columns='index')
    else:
      df = dataframes[year][(dataframes[year]['visit_date'].dt.day == day) & (dataframes[year]['visit_date'].dt.month == month)].reset_index().drop(columns='index')
  elif month!='Not Specified' :
    if site!='Not Specified':
      if weekday!=-1:
        df = dataframes[year][(dataframes[year]['weekday'] == weekday) & (dataframes[year]['visit_date'].dt.month == month) & (dataframes[year]['site'] == site)].reset_index().drop(columns='index')
      else:
        df = dataframes[year][(dataframes[year]['visit_date'].dt.month == month) & (dataframes[year]['site'] == site)].reset_index().drop(columns='index')
    else: 
      if weekday!=-1:
        df = dataframes[year][(dataframes[year]['weekday'] == weekday) & (dataframes[year]['visit_date'].dt.month == month)].reset_index().drop(columns='index')
      else:
        df = dataframes[year][dataframes[year]['visit_date'].dt.month == month].reset_index().drop(columns='index')
  else: 
    if site!='Not Specified':
      if weekday!=-1:
        df = dataframes[year][(dataframes[year]['weekday'] == weekday) & (dataframes[year]['site'] == site)].reset_index().drop(columns='index')
      else:
        df = dataframes[year][dataframes[year]['site'] == site].reset_index().drop(columns='index')
    else:
      if weekday!=-1:
        df = dataframes[year][dataframes[year]['weekday'] == weekday].reset_index().drop(columns='index')
      else:
        df = dataframes[year]
  return df

# function returning the list containing the visit times in decimal scale
def str_to_dectime(day,month,year,weekday,site):
  dectime = []
  strtime = df_by_day_month_year_site_weekday(day,month,year,weekday,site)['visit_time']
  for i in range(len(strtime)):
    dectime.append(float(strtime[i][0:2])+float(strtime[i][3:5])/100*100/60)
  return dectime

#------------------------------------------------------------------------------------------------------
## SIDEBAR

with st.sidebar:
  selected = stm.option_menu("Main Menu", ['Homepage','Data Cleaning','Analysis','Regression'], default_index=0)

#------------------------------------------------------------------------------------------------------
## HOMEPAGE

if selected == 'Homepage':

  st.title('Verona Card - Homepage')

  with st.container():
    st.header('Introduction')
    st.write('''
            VeronaCard is a cumulative ticket released by Comune di Verona, which allows to have a 
            series of advantages for discovering many of the artistic and historical points of interest of 
            our beautiful city, Verona. In particular, it allows to have free or reduced entry to the city main
            museums, monuments and churches together with free city bus travel.
            ''')
    st.write('''
            Here where the advantages are guaranteed:
            ''')
    c1, c2 = st.columns([1,1])
    c1.write('''
              * Arena
              * Arena Museo Opera (AMO)
              * Basilica di San Zeno
              * Basilica di Santa Anastasia
              * Casa di Giulietta
              * Castelvecchio
              * Centro di Fotografia
              * Chiesa di San Fermo
              * Duomo
              * Giardino Giusti
              * Museo Africano
              ''')
    c2.write('''
              * Museo Conte
              * Museo della Radio Epoca
              * Museo di Storia Naturale
              * Museo Lapidario Maffeiano
              * Museo Miniscalchi
              * Palazzo della Ragione            
              * Sightseeing
              * Teatro Romano
              * Tomba di Giulietta
              * Torre dei Lamberti
              * Verona Tour
              ''')
  
  with st.container():
    st.header('Topics')
    st.write('''
            What are the purposes of this project? This project wants to analyze the usage of the VeronaCards 
            in order to reach some conclusions on what is the behaviour of the affluences in different periods of time and different site. 
            More precisely, the purposes of the project are:
            * understanding how affluences changes depending on the **day of the week**
            * understanding how affluences changes depending on the **year**
            * understanding how affluences changes depending on the **month**
            * understanding how affluences changes depending on a **specified date**
            * understanding how affluences changes depending on the **site**
            * foretelling the affluences of 2021 per units of time and per site
            ''')
  
  with st.container():
    st.header('Datasets')
    st.write('''
            In order to analyze the usage of the VeronaCards through the years
            the datasets from 2014 to 2019 has been downloaded from the link: 
            https://dati.veneto.it/catalogo-opendata/comune_di_verona_311.
            ''')  
    st.write('''
            This is how they looks:
            ''')

    # Uploading Datasets
    data_2014 = pd.read_csv('veronacard_2014_opendata.csv')
    data_2015 = pd.read_csv('veronacard_2015_opendata.csv')
    data_2016 = pd.read_csv('veronacard_2016_opendata.csv')
    data_2017 = pd.read_csv('veronacard_2017_opendata.csv')
    data_2018 = pd.read_csv('veronacard_2018_opendata.csv')
    data_2019 = pd.read_csv('veronacard_2019_opendata.csv')

    dataframes = {
      2014: data_2014,
      2015: data_2015,
      2016: data_2016,
      2017: data_2017,
      2018: data_2018,
      2019: data_2019
    }

    year = st.selectbox('Year',list(dataframes.keys()))
    df = dataframes[year]
    st.write('Dataframe dimensions:',str(df.shape[0]),'rows,',str(df.shape[1]),'columns')
    st.dataframe(df)
    st.download_button('Download '+str(year)+' CSV',data = open('veronacard_'+str(year)+'_opendata.csv'),file_name='veronacard_'+str(year)+'_opendata.csv')

#--------------------------------------------------------------------------------------------------------
## DATA CLEANING

if selected == 'Data Cleaning': 
  st.title('Verona Card Utilizzo - Data Cleaning')
  st.write('''
          In order to make an analysis of the VeronaCards usage, the first useful thing to do 
          is cleaning the data. In this way it is possible to keep in memory only what does make 
          sense to be part of the study. Below, it is observable how the datasets have been cleaned step-by-step, 
          in particular the 2014 VeronaCard Dataset.
          ''')

  # Original Dataset
  with st.container():
    st.subheader('Original Dataset')
    st.write('''
            This is how the original dataset is constructed. A notable fact is that some columns are
            useless for the project: ***id_veronacard***, ***profilo***, ***data_attivazione***, ***sito_latitudine***,
            ***sito_longitudine*** has to be dropped.
            ''')
    data_2014 = pd.read_csv('veronacard_2014_opendata.csv')
    st.dataframe(data_2014)
  
  # Useless columns drop
  with st.container():
    st.subheader('Useless columns drop')
    st.write('''
            Some columns are
            useless for the project: ***id_veronacard***, ***profilo***, ***data_attivazione***, ***sito_latitudine***,
            ***sito_longitudine*** are dropped.
            ''')
    data_2014 = data_2014.drop(columns=['data_attivazione','profilo','id_veronacard','sito_latitudine','sito_longitudine'])
    st.dataframe(data_2014)

  # Format column conversion and column addition
  with st.container():
    st.subheader('Format column conversion and column addition')
    st.write('''
            In the previous steps the column data_visita has string-format, which is not the best format in order to be ready 
            to select just one between the day, the month and the year. So, it is advisable to convert that column to a datetime-format. 
            Moreover, because of the project purpose, it is useful to add the weekday column: it represents the day of the week 
            corresponding to the date. 
            ''')
    data_2014['data_visita'] = pd.to_datetime(data_2014['data_visita'])
    data_2014['weekday'] = data_2014['data_visita'].dt.weekday
    st.dataframe(data_2014)

  # Dataset sort and change of columns name
  with st.container():
    st.subheader('Dataset sort and change of columns name')
    st.write('''
            Last but not least, the dataset is sorted by ***visit_date*** and ***visit_time***, after the renomination
            of the columns to the english corresponding names.
            ''')
    data_2014 = data_2014.sort_values(by=['data_visita','ora_visita']).reset_index().drop(columns='index')
    data_2014.rename(columns={'data_visita': 'visit_date', 'ora_visita': 'visit_time', 'sito_nome': 'site'}, inplace=True)
    st.dataframe(data_2014)

#----------------------------------------------------------------------------------------------------
## ANALYSIS
if selected == 'Analysis':

  data_2014 = pd.read_csv('data_2014.csv')
  data_2015 = pd.read_csv('data_2015.csv')
  data_2016 = pd.read_csv('data_2016.csv')
  data_2017 = pd.read_csv('data_2017.csv')
  data_2018 = pd.read_csv('data_2018.csv')
  data_2019 = pd.read_csv('data_2019.csv')

  data_2014.drop(columns='Unnamed: 0',inplace=True)
  data_2015.drop(columns='Unnamed: 0',inplace=True)
  data_2016.drop(columns='Unnamed: 0',inplace=True)
  data_2017.drop(columns='Unnamed: 0',inplace=True)
  data_2018.drop(columns='Unnamed: 0',inplace=True)
  data_2019.drop(columns='Unnamed: 0',inplace=True)

  data_2014['visit_date'] = pd.to_datetime(data_2014['visit_date'])
  data_2015['visit_date'] = pd.to_datetime(data_2015['visit_date'])
  data_2016['visit_date'] = pd.to_datetime(data_2016['visit_date'])
  data_2017['visit_date'] = pd.to_datetime(data_2017['visit_date'])
  data_2018['visit_date'] = pd.to_datetime(data_2018['visit_date'])
  data_2019['visit_date'] = pd.to_datetime(data_2019['visit_date'])

  data_2014_2019 = pd.concat([data_2014,data_2015,data_2016,data_2017,data_2018,data_2019], ignore_index=True)

  weekdays = {
      'Not Specified':-1,
      'Monday':0,
      'Tuesday':1,
      'Wednesday':2,
      'Thursday':3,
      'Friday':4,
      'Saturday':5,
      'Sunday':6
    }

  week = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

  week_2 = ['0_Monday','1_Tuesday','2_Wednesday','3_Thursday','4_Friday','5_Saturday','6_Sunday']

  sites = sorted(list(data_2014_2019['site'].unique()))

  years = range(2014,2020)

  months = range(1,13)

  dataframes = {
      'Not Specified': data_2014_2019,
      2014: data_2014,
      2015: data_2015,
      2016: data_2016,
      2017: data_2017,
      2018: data_2018,
      2019: data_2019
    }

  st.title('Verona Card Utilizzo - Analysis')
  options = st.multiselect('Choose the site to analyze or the sites to compare:',sites,'Arena')

  # By year
  st.header('By year')
  matrix = [[len(df_by_day_month_year_site_weekday('Not Specified','Not Specified',year,-1,site)) for year in years] for site in options]
  df_year = pd.DataFrame(matrix, index = options, columns = years).T
  
  if len(options) == 1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total visits", int(sum(df_year[options[0]])))
    col2.metric("Avg visits per year", round(np.mean(df_year[options[0]]),2))
    col3.metric("Highest number of visits", int(max(df_year[options[0]])))
    col4.metric("Year with most number of visits", 2014+np.argmax(df_year[options[0]]))
    st.bar_chart(df_year)
  else: 
    st.line_chart(df_year)

  # By month
  st.header('By month')
  year = st.selectbox('Year:',['Not Specified']+list(years))
  matrix = [[len(df_by_day_month_year_site_weekday('Not Specified',month,year,-1,site))/len(years) for month in months] for site in options]
  df_month = pd.DataFrame(matrix, index = options, columns = months).T

  if len(options) == 1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg visits per month", round(df_month[options[0]].mean(),2))
    col2.metric("Highest avg number of visits", round(df_month[options[0]].max(),2))
    col3.metric("Month with most avg number of visits", datetime.strptime(str(1+np.argmax(df_month[options[0]])), "%m").strftime("%B"))
    st.bar_chart(df_month)
  else: 
    st.line_chart(df_month)
  
  # By day of the week
  st.header('By day of the week')
  month = st.selectbox('Month:',['Not Specified']+list(months))
  matrix = [[len(df_by_day_month_year_site_weekday('Not Specified',month,year,weekday,site))/len(df_by_day_month_year_site_weekday('Not Specified',month,year,weekday,site).groupby('visit_date')) for weekday in range(0,7)] for site in options]
  df_weekday = pd.DataFrame(matrix, index = options, columns = week_2).T

  if len(options) == 1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg visits per day of the week",round(df_weekday[options[0]].mean(),2))
    col2.metric("Highest avg number of visits", round(df_weekday[options[0]].max(),2))
    col3.metric("Day with most avg number of visits", week[np.argmax(df_weekday[options[0]])])
    st.bar_chart(df_weekday)
  else: 
    st.line_chart(df_weekday)
  
  # By hour
  st.header('By hour')
  options_2 = st.radio('Select:',['Date', 'Period'], index=1)
  if options_2 == 'Date':
    date_selection = st.date_input('Date:', value=min(data_2014_2019['visit_date']), min_value=min(data_2014_2019['visit_date']), max_value=max(data_2014_2019['visit_date']))
    day = date_selection.day
    month = date_selection.month
    year = date_selection.year
    weekday = -1
  else:
    c1, c2, c3 = st.columns([1,1,1])
    day = 'Not Specified'
    month = c1.selectbox('Month:',['Not Specified']+list(months), key = 3, index = (['Not Specified']+list(months)).index(month))
    year = c2.selectbox('Year:',['Not Specified']+list(years), key = 4, index = (['Not Specified']+list(years)).index(year))
    weekday = weekdays[c3.selectbox('Day of the week:',['Not Specified']+week)]

  avg_affluence = [plt.hist(str_to_dectime(day,month,year,weekday,site),bins=np.arange(7.5,20.5))[0]/len(df_by_day_month_year_site_weekday(day,month,year,weekday,site).groupby('visit_date')) for site in options]
  plt.close()
  df_hour = pd.DataFrame(avg_affluence, index = options,columns = range(8,20)).T

  if len(options) == 1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg visits per hour",round(df_hour[options[0]].mean(),2))
    col2.metric("Highest avg number of visits", round(df_hour[options[0]].max(),2))
    col3.metric("Hour with most avg number of visits", 8+np.argmax(df_hour[options[0]]))
    st.bar_chart(df_hour)
  else: 
    st.line_chart(df_hour)

#-------------------------------------------------------------------------------------------------------
## QUARTA PAGINA
if selected == 'Regression':
  st.title('Verona Card Utilizzo - Regression')

  data_2014 = pd.read_csv('data_2014.csv')
  data_2015 = pd.read_csv('data_2015.csv')
  data_2016 = pd.read_csv('data_2016.csv')
  data_2017 = pd.read_csv('data_2017.csv')
  data_2018 = pd.read_csv('data_2018.csv')
  data_2019 = pd.read_csv('data_2019.csv')

  data_2014.drop(columns='Unnamed: 0',inplace=True)
  data_2015.drop(columns='Unnamed: 0',inplace=True)
  data_2016.drop(columns='Unnamed: 0',inplace=True)
  data_2017.drop(columns='Unnamed: 0',inplace=True)
  data_2018.drop(columns='Unnamed: 0',inplace=True)
  data_2019.drop(columns='Unnamed: 0',inplace=True)

  data_2014['visit_date'] = pd.to_datetime(data_2014['visit_date'])
  data_2015['visit_date'] = pd.to_datetime(data_2015['visit_date'])
  data_2016['visit_date'] = pd.to_datetime(data_2016['visit_date'])
  data_2017['visit_date'] = pd.to_datetime(data_2017['visit_date'])
  data_2018['visit_date'] = pd.to_datetime(data_2018['visit_date'])
  data_2019['visit_date'] = pd.to_datetime(data_2019['visit_date'])

  data_2014_2019 = pd.concat([data_2014,data_2015,data_2016,data_2017,data_2018,data_2019], ignore_index=True)

  weekdays = {
      'Not Specified':-1,
      'Monday':0,
      'Tuesday':1,
      'Wednesday':2,
      'Thursday':3,
      'Friday':4,
      'Saturday':5,
      'Sunday':6
    }

  week = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

  week_2 = ['0_Monday','1_Tuesday','2_Wednesday','3_Thursday','4_Friday','5_Saturday','6_Sunday']

  top_sites = ['Arena','Casa Giulietta','Castelvecchio','Duomo','Palazzo della Ragione','Santa Anastasia','Teatro Romano','Tomba Giulietta','Torre Lamberti']

  months = range(1,13)

  # Purposes
  st.header('Purpose')
  st.write('''
          The last part of the project is about prediction.
          As known, 2020 has been one of the most difficult year in modern history since COVID-19 spread all over the world and,
          for this, all the VeronaCard affiliated sites have been closed for several months. In addition to this,
          once the sites reopened to the public, many people that were usually coming to Verona during summer, weren't in that specific year,
          forced by the restrictions. For this reason, the aim of the project is the prediction of the number of visits for each
          day in the year 2020, based on data coming from 2014 to 2019, as if COVID-19 hadn't been there.
          ''')

  # Data Cleaning
  st.header('Data Cleaning')
  st.write('''
          As explained above, the predictions are based on the data coming from 2014 to 2019, but the difference with the previous datasets
          is that only the most renomated and visited sites that have been part of the conventions in all these years are considered.
          The reason behind this decision is the fact that the accuracy of the predictions by doing this is higher.
          These places are:
          * Arena
          * Basilica di Santa Anastasia
          * Casa di Giulietta
          * Castelvecchio
          * Duomo
          * Palazzo della Ragione
          * Teatro Romano
          * Tomba di Giulietta
          * Torre dei Lamberti
          ''')

  # New Dataset
  st.subheader('New Dataset')
  data_14_19_filtered = data_2014_2019[data_2014_2019.site.isin(top_sites)].reset_index().drop(columns='index')  
  st.write('Dataset dimensions:',str(data_14_19_filtered.shape[0]),'observations,',str(data_14_19_filtered.shape[1]),'variables')
  st.dataframe(data_14_19_filtered)

  # Regression  
  st.header('Regression')  
  regression = st.radio('Choose the type of regression:',['Linear Regression', 'Random Forest Regression'])

  data = data_14_19_filtered.groupby(['visit_date','site']).count().reset_index()
  data['weekday'] = data['visit_date'].dt.day_name()
  data = data.rename(columns={'visit_time': 'visits'})
  visits_2014_2019 = data.copy()
  data['day'] = data['visit_date'].dt.day
  data['month'] = data['visit_date'].dt.month
  data['year'] = data['visit_date'].dt.year
  data = data.drop(columns='visit_date')
  data = data[['visits','day','month','year','weekday','site']]
  data = pd.concat([data,pd.get_dummies(data['site']),pd.get_dummies(data['weekday'])], axis=1)
  data = data.drop(columns = ['weekday','site'])

  X = data.drop(columns='visits')
  y = data['visits']
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=2)
  c1,c2 = st.columns(2)
  c1.metric('Linear Regression R^2', round(LinearRegression().fit(X_train,y_train).score(X_test,y_test),2))
  c2.metric('Random Forest Regression R^2', round(RandomForestRegressor(random_state = 1).fit(X_train,y_train).score(X_test,y_test),2))

  if regression == 'Linear Regression':
    reg = LinearRegression().fit(X,y)
  else:
    reg = RandomForestRegressor(random_state = 1).fit(X, y)

  # Train Dataset
  with st.expander('Train Dataset'):
    st.subheader('Train Dataset')
    st.write('Dataset dimensions:',str(data.shape[0]),'observations,',str(data.shape[1]),'variables')
    st.dataframe(data)
  
    fig = plt.figure(figsize=(16, 6))
    sb.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
    st.write(fig)

  # 2020 Dataset
  year_2020 = pd.Series(pd.date_range("2020-1-1", periods=365, freq="D"))
  list_year_2020 = [day for day in year_2020 for i in range(len(top_sites))]

  df_2020 = pd.DataFrame()
  df_2020['visit_date'] = list_year_2020
  df_2020['site'] = top_sites*365
  df_2020['weekday'] = df_2020['visit_date'].dt.day_name()
  df_2020['day'] = df_2020['visit_date'].dt.day
  df_2020['month'] = df_2020['visit_date'].dt.month
  df_2020['year'] = df_2020['visit_date'].dt.year
  df_2020 = pd.concat([df_2020,pd.get_dummies(df_2020['site']),pd.get_dummies(df_2020['weekday'])], axis=1)
  X_df_2020 = df_2020.drop(columns=['visit_date','site','weekday'])
    
  reg_results = reg.predict(X_df_2020)
  X_df_2020['visits'] = reg_results
  X_df_2020 = X_df_2020[['visits']+[col for col in X_df_2020.columns if col != 'visits']]

  df_2020['visits'] = reg_results
  df_2020 = df_2020[['visits']+[col for col in df_2020.columns if col != 'visits']]

  with st.expander('2020 Dataset'):
    st.subheader('2020 Dataset')
    st.write('Dataset dimensions:',str(X_df_2020.shape[0]),'observations,',str(X_df_2020.shape[1]),'variables')
    st.dataframe(X_df_2020)

    fig = plt.figure(figsize=(16, 6))
    sb.heatmap(X_df_2020.corr(), vmin=-1, vmax=1, annot=True)
    st.write(fig)

  # Comparison through years
  st.header('Comparison through years')

  site = st.selectbox('Site:',top_sites)
  visits_2014_2020_site = pd.concat([visits_2014_2019[(visits_2014_2019['site'] == site)], df_2020[df_2020['site'] == site]])[['visit_date','site','visits','weekday']]
  num = round(visits_2014_2020_site.groupby([visits_2014_2020_site['visit_date'].dt.year]).sum()['visits'][2020])
  perc = 100*round((num-visits_2014_2020_site.groupby([visits_2014_2020_site['visit_date'].dt.year]).sum()['visits'][2019])/visits_2014_2020_site.groupby([visits_2014_2020_site['visit_date'].dt.year]).sum()['visits'][2019],2)
  st.metric("2020 visits", num, "{}% (respect 2019)".format(perc))
  st.bar_chart(visits_2014_2020_site.groupby([visits_2014_2020_site['visit_date'].dt.year]).sum())
  
  # Months visits through years
  st.subheader('Months visits through years')
  month = st.selectbox('Month:',list(months))
  a = visits_2014_2020_site[visits_2014_2020_site['visit_date'].dt.month == month]
  month_name = datetime.strptime(str(month), "%m").strftime("%B")
  num = round(a.groupby([a['visit_date'].dt.year]).sum()['visits'][2020])
  perc = 100*round((num-a.groupby([a['visit_date'].dt.year]).sum()['visits'][2019])/a.groupby([a['visit_date'].dt.year]).sum()['visits'][2019],2)
  st.metric("{} visits".format(month_name), num, "{}% (respect 2019)".format(perc))
  st.bar_chart(a.groupby([a['visit_date'].dt.year]).sum())

  # Days of the week visits through years
  st.subheader('Days of the week visits through years')
  weekday = st.selectbox('Day of the week:',week)
  b = a[a['weekday'] == weekday]
  num = round(b.groupby([b['visit_date'].dt.year]).sum()['visits'][2020])
  perc = 100*round((num-b.groupby([b['visit_date'].dt.year]).sum()['visits'][2019])/b.groupby([b['visit_date'].dt.year]).sum()['visits'][2019],2)
  st.metric("{} visits".format(weekday), num, "{}% (respect 2019)".format(perc))
  st.bar_chart(b.groupby([b['visit_date'].dt.year]).sum())

