import pandas as pd
import numpy as np
import seaborn as sns
import regex as re
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta


def format_columns(df):
    '''
    Formatea los nombres de la columnas para estandarizarlas, eliminando espacios y capitalizando.

    Params:
    df: DataFrame
    '''
    lista_columnas=[]
    for i in df.columns:
        lista_columnas.append(i.strip().capitalize())
    df.columns = lista_columnas
    return df.columns


def drop_duplicates(df):
    '''
    Idenfitica la cantidad de nulos existentes en nuestro DataFrame y los elimina.
    
    Params:
    df: DataFrame
    '''
    print(f'Inicialmente hay {df.shape[0]} registros, de los cuales {df.duplicated().sum()} son registros duplicados.')
    print('Eliminando duplicados...')
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    print(f'Tras la limpieza hay {df.shape[0]} registros y hay un total de {df.duplicated().sum()} registros duplicados.')


def case_number_clean(df, column):
    '''
    Función encargada de limpiar la columna 'Case number'

    Params:
    df: DataFrame
    '''
    df['Case number'].fillna('Unknown', inplace=True)
    df['Case number'].astype('str')
    dates=[]

    for i in range(0,df.shape[0]):
        if len(df.iloc[i][column])>=10:
            dates.append(df.iloc[i][column][:10])
        else:
            dates.append('Unknown')

    df[column]=dates
    df[column]=df[column].apply(lambda x: x.replace('.','-'))

    for i in range(0,df.shape[0]):
        try:
             pd.to_datetime(df.iloc[i][column])
        except:
            df.iloc[i][column]=np.nan

    df[column] =  pd.to_datetime(df[column])
    return df[column]


def type_clean(df, column):
    lista_type=[]
    for i in range(0,df.shape[0]):
        if (df.iloc[i][column]!='Unprovoked') and (df.iloc[i][column]!='Provoked') and (df.iloc[i][column]!=np.nan):
            lista_type.append('Undefined')
        else:
            lista_type.append(df.iloc[i][column])
    df[column]=lista_type
    return df[column]


def country_clean(df, column):
    df[column]=df[column].str.strip()
    df[column]=df[column].str.upper()
    df[column].replace(to_replace='[?]', value='', inplace=True, regex=True)

    undefined_list=['EGYPT / ISRAEL','NORTHERN ARABIAN SEA','NORTH ATLANTIC OCEAN',
                'RED SEA / INDIAN OCEAN','PACIFIC OCEAN','NORTH SEA','THE BALKANS',
                'INDIAN OCEAN','IRAN / IRAQ','CENTRAL PACIFIC','SOUTHWEST PACIFIC OCEAN',
                'MID-PACIFC OCEAN','ITALY / CROATIA','WEST INDIES','EQUATORIAL GUINEA / CAMEROON',
                'AFRICA','COAST OF AFRICA','TASMAN SEA','MEDITERRANEAN SEA','RED SEA','ASIA',
                'ATLANTIC OCEAN','PALESTINIAN TERRITORIES','TURKS & CAICOS','CARIBBEAN SEA',
                'SOUTH PACIFIC OCEAN','NORTH PACIFIC OCEAN','OCEAN','BETWEEN PORTUGAL & INDIA',
                'BRITISH WEST INDIES','GULF OF ADEN','MID ATLANTIC OCEAN','PERSIAN GULF','SOUTH ATLANTIC OCEAN',]

    df[column].replace(to_replace=undefined_list, value='UNDEFINED', inplace=True)
    df[column].replace(to_replace=np.nan, value='UNDEFINED', inplace=True)
    df[column].replace('BAY OF BENGAL','INDIA',inplace=True)
    df[column].replace('ANDAMAN / NICOBAR ISLANDAS','INDIA',inplace=True)
    df[column].replace('ANDAMAN ISLANDS','INDIA',inplace=True)
    df[column].replace('NEW GUINEA','PAPUA NEW GUINEA',inplace=True)
    df[column].replace('DIEGO GARCIA','UNITED KINGDOM',inplace=True)
    df[column].replace('CEYLON (SRI LANKA)','SRI LANKA',inplace=True)
    df[column].replace('CEYLON','SRI LANKA',inplace=True)
    df[column].replace('TOBAGO','TRINIDAD & TOBAGO',inplace=True)
    df[column].replace('ST HELENA, BRITISH OVERSEAS TERRITORY','BRITISH VIRGIN ISLANDS',inplace=True)
    df[column].replace('UNITED ARAB EMIRATES (UAE)','UNITED ARAB EMIRATES',inplace=True)
    df[column].replace('SOUTH CHINA SEA','CHINA',inplace=True)
    df[column].replace('WESTERN SAMOA','SAMOA',inplace=True)
    df[column].replace('NEW BRITAIN','PAPUA NEW GUINEA',inplace=True)
    df[column].replace('GUINEA','PAPUA NEW GUINEA',inplace=True)
    df[column].replace('AMERICAN SAMOA','SAMOA',inplace=True)
    df[column].replace('NETHERLANDS ANTILLES','NETHERLANDS',inplace=True)
    df[column].replace('NORTHERN MARIANA ISLANDS','MARIANA ISLANDS',inplace=True)
    df[column].replace('SOLOMON ISLANDS / VANUATU','SOLOMON ISLANDS',inplace=True)
    df[column].replace('REUNION ISLAND','FRANCE',inplace=True)
    df[column].replace('CRETE','GREECE',inplace=True)
    df[column].replace('ADMIRALTY ISLANDS','PAPUA NEW GUINEA',inplace=True)
    df[column].replace('BRITISH NEW GUINEA','PAPUA NEW GUINEA',inplace=True)
    df[column].replace('ANTIGUA','ANTIGUA & BARBUDA',inplace=True)
    df[column].replace('AZORES','PORTUGAL',inplace=True)
    df[column].replace('BRITISH ISLES','UNITED KINGDOM',inplace=True)
    df[column].replace('SCOTLAND','UNITED KINGDOM',inplace=True)
    df[column].replace('ENGLAND','UNITED KINGDOM',inplace=True)
    df[column].replace('COLUMBIA','COLOMBIA',inplace=True)
    df[column].replace('GRAND CAYMAN','CAYMAN ISLANDS',inplace=True)
    df[column].replace('GUAM','USA',inplace=True)
    df[column].replace('HONG KONG','CHINA',inplace=True)
    df[column].replace('MALDIVES','MALDIVE ISLANDS',inplace=True)
    df[column].replace('OKINAWA','JAPAN',inplace=True)
    df[column].replace('REUNIION','HONDURAS',inplace=True)
    df[column].replace('ROATAN','HONDURAS',inplace=True)
    df[column].replace('SAN DOMINGO','DOMINICAN REPUBLIC',inplace=True)
    df[column].replace('SOUTH KOREA','KOREA',inplace=True)
    df[column].replace('ST. MAARTIN','SAINT MARTIN',inplace=True)
    df[column].replace('ST. MARTIN','SAINT MARTIN',inplace=True)
    return df[column].value_counts()


def activity_clean(df, column):
    df[column]=df[column].fillna('non common activity')
    df[column]=[re.sub(r'[^\w\s]','',str(i)) for i in df[column]]
    df[column]=df[column].str.strip()
    df[column]=df[column].str.lower()

    activities=[]
    for i in df[column].value_counts()[:25].index:
        activities.append(i)

    list_activity=[]

    check=False
    for i in range(0,df.shape[0]):
        for e in activities:
            if re.search(e, str(df.iloc[i][column])):
                list_activity.append(e)
                check=True
                break
        if check == False:
            list_activity.append('non common activity')
        check = False
    
    df[column]=list_activity

    df[column].replace('diving','diving activity',inplace=True)
    df[column].replace('boogie boarding','body boarding',inplace=True)
    df[column].replace('canoeing','kayaking',inplace=True)

    return df[column].value_counts()


def sex_clean(df, column):
    df[column]=df[column].str.strip()
    lista_sex=[]
    for i in range(0,df.shape[0]):
        if (df.iloc[i][column]!='M') and (df.iloc[i][column]!='F'):
            lista_sex.append(np.nan)
        else:
            lista_sex.append(df.iloc[i][column])
    df[column]=lista_sex

    df[column].replace(to_replace=np.nan,value='Not defined',inplace=True)

    return df[column].value_counts()


def age_clean(df, column):
    df[column]=df[column].str.strip()
    df[column]=pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
    df[column].replace(0, np.nan, inplace=True)

    return df[column].value_counts()


def fatal_clean(df, column):
    df[column]=df[column].str.strip()
    df[column]=df[column].str.upper()

    lista_fatal=[]
    for i in range(0,df.shape[0]):
        if (df.iloc[i][column]!='N') and (df.iloc[i][column]!='Y'):
            lista_fatal.append('UNKNOWN')
        else:
            lista_fatal.append(df.iloc[i][column])

    df[column]=lista_fatal
    return df[column].value_counts()


def species_clean(df, column):
    df[column].fillna('unknown', inplace=True)
    df[column]=[re.sub(r'[^\w\s]','',str(i)) for i in df[column]]
    df[column]=df[column].str.strip()
    df[column]=df[column].str.lower()

    shark_species = ['white shark','tiger shark','bull shark','wobbegong shark','blacktip shark','blue shark','mako shark',
             'raggedtooth shark','nurse shark','cooper shark','hammerhead shark','sandtiger shark','lemon shark',
             'spinner shark','angel shark','sevengill shark','galapagos shark']
    list_species=[]

    check=False
    for i in range(0,df.shape[0]):
        for e in shark_species:
            if re.search(e, str(df.iloc[i][column])):
                list_species.append(e)
                check=True
                break
        if check == False:
            list_species.append('not identified')
        check = False
    
    df[column]=list_species

    return df[column].value_counts()


def injury_clear(df, column):
    df[column].fillna('unknown', inplace=True)
    df[column]=[re.sub(r'[^\w\s]','',str(i)) for i in df[column]]
    df[column]=df[column].str.strip()
    df[column]=df[column].str.lower()

    injury_list=['no injury','leg','ankle','foot','calf','arm','hand','heel','knee','elbow','shoulder','torso']
    
    injuries=[]

    check=False
    for i in range(0,df.shape[0]):
        for e in injury_list:
            if re.search(e, str(df.iloc[i][column])):
                injuries.append(e)
                check=True
                break
        if check == False:
            injuries.append('not defined')
        check = False
    
    df[column]=injuries

    return df[column].value_counts()


'''
MÉTODO ALTERNATIVO PARA LIMPIAR COLUMNA 'SPECIES'
# Documento txt con los nombres de todas las especies de tiburones.
# Esta estructurado como un JSON en forma de diccionarios.
# Limpieza por las identaciones.

fi = open('shark-species.txt', 'r')
# The file shark-species.txt has order names starting in the first column,
# family indented by 4 spaces, genus indented by 8 spaces and
# binomial : common name indented by 12 spaces.
sharks_dict = {}
for line in fi.readlines():
    if line.startswith(' '*12):
        # binomial name : common name
        binomial, common_name = line.strip().split(' : ')
        # refactor the binomial name to standard abbreviated form
        species = '{}. {}'.format(genus[0], binomial.split()[1])
        sharks_dict[order][family][genus][species] = common_name
    elif line.startswith(' '*8):
        # A new GENUS: start a new dictionary in its name
        genus = line.strip()
        sharks_dict[order][family][genus] = {}
    elif line.startswith(' '*4):
        # A new FAMILY: start a new dictionary in its name
        family = line.strip()
        sharks_dict[order][family] = {}
    else:
        # A new ORDER: start a new dictionary in its name
        order = line.strip()
        sharks_dict[order] = {}
fi.close()


# sharks_dict  -> Para visualizar el diccionario
# sharks_dict['Lamniformes']['Lamnidae']['Carcharodon']['C. carcharias']  -> Acceder a su información

# Extracción de los nombres y adición en una lista vacía.
list_shark_species=[]
for i in sharks_dict:
    for e in sharks_dict[i]:
        for n in sharks_dict[i][e]:
            for s in sharks_dict[i][e][n]:
                list_shark_species.append(sharks_dict[i][e][n][s].lower())

list_species_2=[]

check=False
for i in range(0,sharks.shape[0]):
    for e in list_shark_species:
        if re.search(e, str(sharks.iloc[i]['Species'])):
            list_species_2.append(e)
            check=True
            break
    if check == False:
        list_species_2.append(np.nan)
    check = False
    
sharks['Species']=list_species_2
sharks['Species'].value_counts()
'''