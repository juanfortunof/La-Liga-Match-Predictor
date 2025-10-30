import pandas as pd
import unicodedata
import time
import random

def clean_text(text):
    
    nfkd_form = unicodedata.normalize('NFKD', text)
    only_ascii_chars = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii_chars

def get_data(seasons):

    hidden_urls = {
    "Barcelona": "206d90db",                     
    "Real-Madrid": "53a2f082",                   
    "Atletico-Madrid": "db3b9613",               
    "Valencia": "dcc91a7b",                      
    "Sevilla": "ad2be733",                       
    "Villarreal": "2a8183b3",                    
    "Real-Sociedad": "e31d1cd9",                 
    "Athletic-Club": "2b390eca",                 
    "Celta-Vigo": "f25da7fb",                    
    "Real-Betis": "fc536746",                    
    "Espanyol": "a8661628",                      
    "Getafe": "7848bd64",                        
    "Levante": "9800b6a1",                       
    "Malaga": "1c896955",                        
    "Girona": "9024a00a",                        
    "Alaves": "8d6fd021",                        
    "Eibar": "bea5c710",                         
    "Deportivo-La-Coruna": "2a60ed82",           
    "Las-Palmas": "0049d422",                    
    "Granada": "a0435291",                       
    "Mallorca": "2aa12281",                      
    "Leganes": "7c6f2c78",                       
    "Valladolid": "17859612",                    
    "Cadiz": "ee7c297c",                         
    "Elche": "6c8b07df",                         
    "Osasuna": "03c57e2b",                       
    "Rayo-Vallecano": "98e8af82",                
    "Huesca": "c6c493e6",                        
    "Almeria": "78ecf4bb",
    "Sporting-Gijon": "bb9efd50",
    "Oviedo": "ab358912"                      
    }

    all_dfs = []

    time.sleep(10)

    for season in seasons:

        teams = [clean_text(team) for team in teams]
        teams = [team.replace(' ', '-') for team in teams]

        for team in teams:

            print(team)

            if team == 'La-Coruna':
                team = 'Deportivo-La-Coruna'
            elif team == 'Betis':
                team = 'Real-Betis'


            scores_url = f'https://fbref.com/en/squads/{hidden_urls[team]}/{str(int(season) - 1)}-{season}/{team}-Stats'
            scores_df = pd.read_html(scores_url)[1]
            scores_df['Team'] = team
            scores_df['season'] = int(season)
            
            time.sleep(random.randint(10, 20))

            shooting_url = f'https://fbref.com/en/squads/{hidden_urls[team]}/{str(int(season) - 1)}-{season}/matchlogs/all_comps/shooting/{team}-Match-Logs-All-Competitions'
            shooting_df = pd.read_html(shooting_url)[0]
            shooting_df = shooting_df.droplevel(level=0, axis=1)
            cols_merge = ['Date', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt']

            for col in cols_merge:
                if col not in shooting_df.columns:
                    shooting_df[col] = 0

            complete_df = scores_df.merge(shooting_df[cols_merge], on='Date')
            complete_df = scores_df
            complete_df = complete_df[complete_df['Comp'] == 'La Liga']

            all_dfs.append(complete_df)
            print('Done')

            time.sleep(random.randint(10, 20))

    return all_dfs


def test_complete_data(length):

    if length == 6080:
        print('Perfect! there is no missing data.')
    else:
        print('There is missing data, the data should have should have 6080 rows')



