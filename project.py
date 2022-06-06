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

#---------------------------------------------------------------------------------------------------
## FUNZIONI

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

# function returning the histogram of the visit times, given a date/period and a site
def plot_affluence_on_time(day,month,year,weekday,site):
  plt.figure()
  plt.hist(str_to_dectime(day,month,year,weekday,site),bins=np.arange(7.5,20.5),rwidth=0.7)
  plt.xticks(np.arange(8,20))
  plt.show()
  plt.xlabel('Hour')
  plt.ylabel('Visits')
  return 

# function returning the histogram of the number of strisciate per days of the week, given a month, a year and a site
def plot_affluence_on_week(month,year,site):
  df = df_by_day_month_year_site_weekday('Not Specified',month,year,-1,site)
  weekdays = df['weekday']
  plt.figure()
  plt.hist(weekdays,bins=np.arange(-0.5,7.5),rwidth=0.7)
  plt.xticks(np.unique(weekdays.values),week,rotation=45)
  plt.xlabel('Week')
  plt.ylabel('Visits')
  plt.show()
  return 

# function returning the histogram of the AVG visit times, given a date/period and a site
def plot_avg_affluence_on_time(day,month,year,weekday,site):
  df = df_by_day_month_year_site_weekday(day,month,year,weekday,site)
  avg_affluence = plt.hist(str_to_dectime(day,month,year,weekday,site),bins=np.arange(7.5,20.5))[0]/len(df.groupby('visit_date'))
  plt.close()
  plt.bar(np.arange(8,20),avg_affluence,width=0.7)
  plt.xticks(np.arange(8,20))
  plt.xlabel('Hour')
  plt.ylabel('Visits')
  plt.show()
  return

# function returning the histogram of the AVG number of strisciate per days of the week, given a month, a year and a site
def plot_avg_affluence_on_week(month,year,site):
  df = df_by_day_month_year_site_weekday('Not Specified',month,year,-1,site)
  a = (pd.Series(df.groupby('visit_date').count().index).dt.weekday).value_counts().sort_index()
  plt.bar(week,df['weekday'].value_counts().sort_index()/a,width=0.6)
  plt.xticks(rotation=45)
  plt.xlabel('Week')
  plt.ylabel('Visits')
  plt.show()
  return

#------------------------------------------------------------------------------------------------------
## SIDEBAR

with st.sidebar:
    selected = stm.option_menu("Main Menu", ['Homepage','Data Cleaning','Analysis','Comparison','Regression'], default_index=0)

#---------------------------------------------------------------------------------------------
## PRIMA PAGINA

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
            the datasets from 2014 to 2020 has been downloaded from the link: 
            https://dati.veneto.it/catalogo-opendata/comune_di_verona_311.
            ''')  
    st.write('''
            This is how they looks:
            ''')

    ## CARICAMENTO DEI DATASETS
    data_2014 = pd.read_csv('veronacard_2014_opendata.csv')
    data_2015 = pd.read_csv('veronacard_2015_opendata.csv')
    data_2016 = pd.read_csv('veronacard_2016_opendata.csv')
    data_2017 = pd.read_csv('veronacard_2017_opendata.csv')
    data_2018 = pd.read_csv('veronacard_2018_opendata.csv')
    data_2019 = pd.read_csv('veronacard_2019_opendata.csv')

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

    week=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    dataframes = {
      #'Not Specified': data_2014_2019,
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
## SECONDA PAGINA

if selected == 'Data Cleaning': 
  st.title('Verona Card Utilizzo - Data Cleaning')
  st.write('''
          In order to make an analysis of the VeronaCards usage, the first useful thing to do 
          is cleaning the data. In this way it is possible to keep in memory only what does make 
          sense to be part of the study. Below, it is observable how the datasets have been cleaned step-by-step, 
          in particular the 2014 VeronaCard Dataset.
          ''')

  # DATASET ORIGINALI
  with st.container():
    st.subheader('Original Dataset')
    st.write('''
            This is how the original dataset is constructed. A notable fact is that some columns are
            useless for the project: ***id_veronacard***, ***profilo***, ***data_attivazione***, ***sito_latitudine***,
            ***sito_longitudine*** has to be dropped.
            ''')
    data_2014 = pd.read_csv('veronacard_2014_opendata.csv')
    st.dataframe(data_2014)
  
  # RIMOSSIONE COLONNE INUTILI
  with st.container():
    st.subheader('Useless columns drop ')
    st.write('''
            Some columns are
            useless for the project: ***id_veronacard***, ***profilo***, ***data_attivazione***, ***sito_latitudine***,
            ***sito_longitudine*** are dropped.
            ''')
    data_2014 = data_2014.drop(columns=['data_attivazione','profilo','id_veronacard','sito_latitudine','sito_longitudine'])
    st.dataframe(data_2014)

  # RENDO LE DATE IN FORMATO DATETIME e AGGIUNGO LA COLONNA WEEKDAY
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

  # ORDINO I DATAFRAME PER DATA E ORA e CAMBIO NOME DELLE COLONNE
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
## TERZA PAGINA

if selected == 'Analysis':
  st.title('Verona Card Utilizzo - Analysis')

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

  dataframes = {
      'Not Specified': data_2014_2019,
      2014: data_2014,
      2015: data_2015,
      2016: data_2016,
      2017: data_2017,
      2018: data_2018,
      2019: data_2019
    }

  with st.container():
    st.header('Analysis by year')

    matrix = np.zeros((len(sites),len(years)))
    for i in range(len(sites)):
      for j in range(len(years)):
        matrix[i][j] = len(dataframes[years[j]][dataframes[years[j]]['site']==sites[i]])
    df_year = pd.DataFrame(matrix, columns = years, index = sites).T
    df_year['Not Specified'] = [sum(df_year.T[year]) for year in years]

    site = st.selectbox('Site input',['Not Specified']+sites)

    st.write('**Total number of visits**:', str(int(sum(df_year[site]))))
    st.write('**Average number of visits per year**:', str(round(np.mean(df_year[site]),2)))
    st.write('**Year with the highest number of visits**:', str(2014+np.argmax(df_year[site])))
    st.write('**Highest number of visits**:', str(int(max(df_year[site]))))

    st.bar_chart(df_year[site])

  with st.container():
    st.header('Analysis by month')
    year = st.selectbox('Year input',['Not Specified']+list(years))

    if year != 'Not Specified':
      matrix = [len(df_by_day_month_year_site_weekday('Not Specified',month,year,-1,site)) for month in range(1,13)]
      df_month = pd.DataFrame(matrix, index = range(1,13), columns = [site])
      st.write('**Total number of visits**:', str(int(sum(df_month[site]))))
      st.write('**Average number of visits per month**:', str(round(df_month[site].mean(),2)))
      st.write('**Month with the highest absolute number of visits**:', str(1+np.argmax(df_month[site])))
      st.write('**Highest number of visits**:', str(max(df_month[site])))
      st.bar_chart(df_month)
    else:
      matrix = [[len(df_by_day_month_year_site_weekday('Not Specified',month,year,-1,site)) for month in range(1,13)] for year in years]
      df_month = pd.DataFrame(matrix, index = years, columns = range(1,13))
      st.write('**Average number of visits per month**:', str(round(df_month.mean().mean(),2)))
      st.write('**Month with the highest average number of visits**:', str(1+np.argmax(df_month.mean())) )
      st.write('**Highest average number of visits**:', str(round(df_month.mean().max(),2)))

      options = st.radio('Select:',['Avg visits per month', 'Absolute visits per month'])
      if options == 'Avg visits per month':
        st.bar_chart(pd.DataFrame(df_month.mean(),columns=[site]))
      else: 
        st.bar_chart(pd.DataFrame(df_month.sum(),columns=[site]))

  with st.container():
    st.header('Analysis by day of the week')
    c1, c2 = st.columns([1,1])
    month = c1.selectbox('Month Input',['Not Specified']+list(range(1, 13)), key = 1)
    year = c2.selectbox('Year Input',list(dataframes.keys()), key = 2)

    matrix = [len(df_by_day_month_year_site_weekday('Not Specified',month,year,weekday,site))/len(df_by_day_month_year_site_weekday('Not Specified',month,year,weekday,site).groupby('visit_date')) for weekday in range(0,7)]
    
    st.write('**Average number of visits per day of the week**:', str(round(np.mean(matrix),2)))
    st.write('**Day of the week with the highest average number of visits**:', week[np.argmax(matrix)])
    st.write('**Highest average number of visits**:', str(round(np.max(matrix),2)))

    options = st.radio('Select:',['Avg visits per day of the week', 'Absolute visits per day of the week'])
    if options == 'Absolute visits per day of the week':
      matrix = [len(df_by_day_month_year_site_weekday('Not Specified',month,year,weekday,site)) for weekday in range(0,7)]
    
    df_week = pd.DataFrame(matrix, columns = [str(month)+'/'+str(year)])#, index = week)
    st.bar_chart(df_week)


  with st.container():
    st.header('Analysis by hour')

    options = st.radio('Select:',['By date', 'By period'])
    if options == 'By date':
      date_selection = st.date_input('Date', value=min(data_2014_2019['visit_date']), min_value=min(data_2014_2019['visit_date']), max_value=max(data_2014_2019['visit_date']))
      day = date_selection.day
      month = date_selection.month
      year = date_selection.year
      weekday = -1
    else:
      c1, c2, c3 = st.columns([1,1,1])
      day = 'Not Specified'
      month = c1.selectbox('Month',['Not Specified']+list(range(1, 13)),key = 1)
      year = c2.selectbox('Year',list(dataframes.keys()),key = 2)
      weekday = weekdays[c3.selectbox('Day of the week',['Not Specified']+week)]
  
    df = df_by_day_month_year_site_weekday(day,month,year,weekday,site)
    avg_affluence = plt.hist(str_to_dectime(day,month,year,weekday,site),bins=np.arange(7.5,20.5))[0]/len(df.groupby('visit_date'))
    plt.close()
    df_hour = pd.DataFrame(index = range(8,20))
    df_hour[site] = avg_affluence

    st.write('**Average number of visits per hour**:', str(round(df_hour[site].mean(),2)))
    st.write('**Hour with the highest average number of visits**:', str(8+np.argmax(df_hour[site])))
    st.write('**Highest average number of visits**:', str(round(df_hour[site].max(),2)))

    st.bar_chart(df_hour)

  with st.container():
    st.header('Comparison')
    if site != 'Not Specified':
      options = st.multiselect('Choose the sites to make the comparison',sites,site)
    else:
      options = st.multiselect('Choose the sites to make the comparison',sites,sites[0])
    button_sent = st.button("SUBMIT")

    if button_sent:
      st.line_chart(df_year[options])
  


#-------------------------------------------------------------------------------------------------------
## QUARTA PAGINA
if selected == 'Comparison':

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

  st.title('Verona Card Utilizzo - Comparison')
  options = st.multiselect('Choose the sites to make the comparison',sites,sites[0])

  matrix = [[len(df_by_day_month_year_site_weekday('Not Specified','Not Specified',year,-1,site)) for year in years] for site in options]
  df = pd.DataFrame(matrix, index = options, columns = years).T
  st.line_chart(df)

  year = st.selectbox('Year',['Not Specified']+list(years))
  matrix = [[len(df_by_day_month_year_site_weekday('Not Specified',month,year,-1,site))/len(years) for month in months] for site in options]
  df = pd.DataFrame(matrix, index = options, columns = months).T
  st.line_chart(df)

  c1, c2 = st.columns([1,1])
  month = c1.selectbox('Month Input',['Not Specified']+list(months), key = 1)
  year = c2.selectbox('Year',['Not Specified']+list(years), key = 2)
  matrix = [[len(df_by_day_month_year_site_weekday('Not Specified',month,year,weekday,site))/len(df_by_day_month_year_site_weekday('Not Specified',month,year,weekday,site).groupby('visit_date')) for weekday in range(0,7)] for site in options]
  df = pd.DataFrame(matrix, index = options, columns = week_2).T
  st.line_chart(df)

  options_2 = st.radio('Select:',['By date', 'By period'])
  if options_2 == 'By date':
    date_selection = st.date_input('Date', value=min(data_2014_2019['visit_date']), min_value=min(data_2014_2019['visit_date']), max_value=max(data_2014_2019['visit_date']))
    day = date_selection.day
    month = date_selection.month
    year = date_selection.year
    weekday = -1
  else:
    c1, c2, c3 = st.columns([1,1,1])
    day = 'Not Specified'
    month = c1.selectbox('Month',['Not Specified']+list(months), key = 1)
    year = c2.selectbox('Year',['Not Specified']+list(years), key = 2)
    weekday = weekdays[c3.selectbox('Day of the week',['Not Specified']+week)]

  avg_affluence = [plt.hist(str_to_dectime(day,month,year,weekday,site),bins=np.arange(7.5,20.5))[0]/len(df_by_day_month_year_site_weekday(day,month,year,weekday,site).groupby('visit_date')) for site in options]
  plt.close()
  df_hour = pd.DataFrame(avg_affluence, index = options,columns = range(8,20)).T
  st.line_chart(df)

#-------------------------------------------------------------------------------------------------------
## QUINTA PAGINA
if selected == 'Regression':
  st.title('Verona Card Utilizzo - Regression')









#https://dati.veneto.it/content/dati_veronacard_2014
#https://dati.veneto.it/content/dati_veronacard_2015
#https://dati.veneto.it/content/dati_veronacard_2016
#https://dati.veneto.it/content/dati_veronacard_2017
#https://dati.veneto.it/content/dati_veronacard_2018
#https://dati.veneto.it/content/dati_veronacard_2019
#https://dati.veneto.it/content/dati_veronacard_2020
