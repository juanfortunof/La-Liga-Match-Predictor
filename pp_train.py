import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class form_clean_transf:

    def __init__(self, data, formation_col):
        self.data = data
        self.formation_col = formation_col

    def clean_formation_col(self, new_data: pd.DataFrame):
        try:
            new_data[self.formation_col] = self.data[self.formation_col].str.replace('◆', '')
            new_data[self.formation_col] = self.data[self.formation_col].replace('4-2-4-0', '4-2-4')
        except Exception as e:
            print('Error while cleaning the formation column.')
            raise e

        return new_data

    def separate_formation(self, new_data: pd.DataFrame):

        try:
            new_data['formation_defenders'] = self.data[self.formation_col].str.split('-').str[0]
            new_data['formation_midfielders'] = self.data[self.formation_col].str.split('-').str[1:-1].apply(lambda l: sum(int(num) for num in l))
            new_data['formation_strikers'] = self.data[self.formation_col].str.split('-').str[-1]

        except Exception as e:
            print('Error on the separate formation method')
            raise e    

        return new_data
    

class month_transf:

    def __init__(self, data, date_col):
        self.data = data
        self.date_col = date_col

    @staticmethod
    def __first_part(day):
        if day < 16:
            return 1
        else:
            return 0
    
    @staticmethod
    def __second_part(day):
        if day > 15:
            return 1
        else:
            return 0

    def first_sec_part_and_month(self, new_data):

        try:
            self.data['date'] = pd.to_datetime(self.data[self.date_col])
            
            new_data['month'] = self.data[self.date_col].dt.month
            new_data['played_first_part_of_month'] = self.data[self.date_col].dt.day.apply(self.__first_part)
            new_data['played_second_part_of_month'] = self.data[self.date_col].dt.day.apply(self.__second_part)

        except Exception as e:
            print('Error at first_sec_part_month')
            raise e

        return new_data    


class round_transf:

    def __init__(self, data, round_col):
        self.data = data
        self.round_col = round_col

    def encode_col(self, new_data):
        try:
            new_data['round_encoded'] = self.data[self.round_col].str.split(' ').str[-1].astype(int)
        except Exception as e:
            print('Error at encoding the round col')
            raise e
        
        return new_data

    def is_first_round(self, new_data):
        try:
            new_data['is_first_round'] = (new_data['round_encoded'] <= 19).astype(int)
        except Exception as e:
            print('Error at round first round func')
            raise e
          
        return new_data

    def is_second_round(self, new_data):
        try:
            new_data['is_second_round'] = (new_data['round_encoded'] > 19).astype(int)
        except Exception as e:
            print('Error at round second round func')
            raise e
        
        return new_data
    
    
class one_hot_encoder:

    def __init__(self, data: pd.DataFrame, cols: list):
        self.data = data
        self.cols = cols

    def encode(self, new_data):

        one_hot_df = pd.DataFrame()

        try:
            for col in self.cols:
                one_hot_df = pd.get_dummies(self.data[col], prefix=col).astype(int)
                new_data = pd.concat([new_data, one_hot_df], axis=1)

        except Exception as e:
            print('Error at one hot encoding')
            raise e

        return new_data
    

class column_encoder:

    team_codes = {  'Barcelona': 1,
                'Real Madrid': 2,
                'Atletico Madrid': 3,
                'Valencia': 4,
                'Sevilla': 5,
                'Villarreal': 6,
                'Athletic Club': 7,
                'Celta Vigo': 8,
                'Malaga': 9,
                'Espanyol': 10,
                'Real Sociedad': 11,
                'Levante': 12,
                'Getafe': 13,
                'Real Betis': 14,
                'Eibar': 15,
                'Girona': 16,
                'Alaves': 17,
                'Leganes': 18,
                'Deportivo La Coruna': 19,
                'Las Palmas': 20,
                'Valladolid': 21,
                'Huesca': 22,
                'Rayo Vallecano': 23,
                'Granada': 24,
                'Osasuna': 25,
                'Mallorca': 26,
                'Cadiz': 27,
                'Elche': 28,
                'Almeria': 29,
                'Oviedo': 30}

    def __init__(
                self, 
                data: pd.DataFrame, 
                result_col: str, 
                home_team_col: str, 
                opp_team_col: str):
        self.data = data
        self.result_col = result_col
        self.home_team_col = home_team_col
        self.opp_team_col = opp_team_col


    @staticmethod
    def __static_encode_target(outcome):
        match outcome:
            case 'W': return 2
            case 'D': return 1
            case 'L': return 0

    def encode_target(self, new_data):
        try:
            new_data[self.result_col] = self.data[self.result_col].apply(self.__static_encode_target)
        except Exception as e:
            print("Error while encoding result column")
            raise e
        
        return new_data
    
    def encode_team(self, new_data):
        try:
            new_data['home_team_encoded'] = self.data[self.home_team_col].map(column_encoder.team_codes)
            new_data['opponent_team_encoded'] = self.data[self.opp_team_col].map(column_encoder.team_codes)
        except Exception as e:
            print('Error at the team encoding method')
            raise e
        return new_data
    

class feature_engineer:

    def __init__(self, 
                 data: pd.DataFrame,
                 xg_col: str, 
                 xga_col: str, 
                 poss_col: str, 
                 gf_col: str,
                 ga_col: str, 
                 sot_col: str, 
                 sh_col: str):

        self.data = data
        self.xg_col = xg_col
        self.xga_col = xga_col
        self.poss_col = poss_col
        self.gf_col = gf_col
        self.ga_col = ga_col
        self.sot_col = sot_col 
        self.sh_col = sh_col

    @staticmethod
    def __calculation(num, den):
        if num == 0 or den == 0:
            return 0
        else:
            return num / den
        
    def new_eng_cols(self, new_data):
        try:
            new_data['xg_diff'] = self.data[self.xg_col] - self.data[self.xga_col]
            new_data['poss_a'] = 100 - self.data[self.poss_col]
            new_data['poss_diff'] = self.data[self.poss_col] - new_data['poss_a']
            new_data['effective_possesion'] = self.data[self.xg_col] * self.data[self.poss_col]
            new_data['vulnerable_possesion'] = new_data['poss_a'] * self.data[self.xga_col]
            new_data['gf_ga_diff'] = self.data[self.gf_col] - self.data[self.ga_col]
            new_data['shooting_scoring_efficiency'] = self.data.apply(lambda x: self.__calculation(x[self.gf_col], x[self.sot_col]), axis=1)
            new_data['shooting_efficiency'] = self.data.apply(lambda x: self.__calculation(x[self.sot_col], x[self.sh_col]), axis=1)
        
        except Exception as e:
            print('Error while feature engineering')
            raise e
        
        cont_cols = ['gf', 'ga', 'xg', 'xga', 'poss', 'sot', 'sh', 'dist', 'fk', 'pk', 'pkatt',
                    'xg_diff', 'poss_a', 'poss_diff', 'effective_possesion', 'vulnerable_possesion', 
                    'gf_ga_diff', 'shooting_scoring_efficiency', 'shooting_efficiency']
        
        return new_data, cont_cols
    

class ELO:

    def __init__(self, data, date_col, team_col, opp_col, gf_col, ga_col, season_col, home_col):
        self.data = data
        self.date_col = date_col
        self.team_col = team_col
        self.opp_col = opp_col
        self.gf_col = gf_col
        self.ga_col = ga_col
        self.season_col = season_col
        self.home_col = home_col

    def calculate_ELO(self, new_data):

        elo_history = []
        elo = {team: 1500 for team in new_data[self.team_col].unique().tolist()}
        home_adv = 100
        K = 20

        for idx, row in new_data.iterrows():
            home, away = row[self.team_col], row[self.opp_col]
            elo_home, elo_away = elo[home], elo[away]

            expected_home = 1 / (1 + 10 ** (((elo_away - home_adv) - elo_home) / 400))

            if row[self.gf_col] > row[self.ga_col]:
                score_home = 1
            elif row[self.gf_col] == row[self.ga_col]:
                score_home = 0.5
            else:
                score_home = 0

            delta = K * (score_home - expected_home)
            elo[home] += delta
            elo[away] -= delta

            elo_history.append({
                'date': row[self.date_col],
                'home_team': home,
                'away_team': away,
                'elo_home': elo_home,
                'elo_away': elo_away,
                'elo_diff': elo_home - elo_away
            })

        elos_df = pd.DataFrame(elo_history)

        return elos_df


class full_preprocess:

    def __init__(
                self, 
                data: pd.DataFrame, 
                formation_col: str, 
                date_col: str, 
                round_col: str, 
                one_hot_cols: list,
                result_col: list,
                home_team_col: str,
                opp_team_col: str,
                xg_col: str, 
                xga_col: str, 
                poss_col: str, 
                gf_col: str,
                ga_col: str, 
                sot_col: str, 
                sh_col: str):
        
        self.data = data
        self.formation_col = formation_col
        self.date_col = date_col
        self.round_col = round_col
        self.one_hot_cols = one_hot_cols
        self.result_col = result_col
        self.home_team_col = home_team_col
        self.opp_team_col = opp_team_col
        self.xg_col = xg_col
        self.xga_col = xga_col
        self.poss_col = poss_col
        self.gf_col = gf_col
        self.ga_col = ga_col
        self.sot_col = sot_col 
        self.sh_col = sh_col
   
    def run(self):

        # Create a new empty DataFrame where we will add the new columns

        new_data = self.data.copy()
        form_obj = form_clean_transf(self.data, self.formation_col)
        new_data = form_obj.clean_formation_col(new_data)

        month_obj = month_transf(self.data, self.date_col)
        new_data = month_obj.first_sec_part_and_month(new_data)

        round_obj = round_transf(self.data, self.round_col)
        new_data = round_obj.encode_col(new_data)
        new_data = round_obj.is_first_round(new_data)
        new_data = round_obj.is_second_round(new_data)

        one_hot_encoder_obj = one_hot_encoder(self.data, self.one_hot_cols)
        new_data = one_hot_encoder_obj.encode(new_data)

        column_encoder_obj = column_encoder(self.data, self.result_col, 
                                            self.home_team_col, self.opp_team_col)
        new_data = column_encoder_obj.encode_target(new_data)
        new_data = column_encoder_obj.encode_team(new_data)

        new_data = form_obj.separate_formation(new_data)
       
        feature_engineer_obj = feature_engineer(self.data, 'xg', 'xga', 'poss', 'gf', 'ga', 'sot', 'sh')
        new_data, _ = feature_engineer_obj.new_eng_cols(new_data)

        elo_obj = ELO(new_data, 'date', 'team', 'opponent', 'gf', 'ga', 'season', 'venue_Home')

        new_data = new_data[new_data['venue_Home'] == 1].sort_values(by=self.date_col).reset_index().drop(columns='index')

        elos_dfs = []

        for season in self.data['season'].unique().tolist():
            temp_data = elo_obj.calculate_ELO(new_data[new_data['season'] == season])
            elos_dfs.append(temp_data)

        merged_elos = pd.concat(elos_dfs).reset_index().drop(columns='index')

        new_data = new_data.join(merged_elos[['elo_home', 'elo_away', 'elo_diff']])

        cont_cols = ['gf', 'ga', 'xg', 'xga', 'poss', 'sot', 'sh', 'dist', 'fk', 'pk', 'pkatt',
                    'xg_diff', 'poss_a', 'poss_diff', 'effective_possesion', 'vulnerable_possesion', 
                         'gf_ga_diff', 'shooting_scoring_efficiency', 'shooting_efficiency', 'elo_home',
                         'elo_away', 'elo_diff']

        cat_cols = ['result', 'month', 'played_first_part_of_month', 'played_second_part_of_month',
                    'round_encoded', 'is_first_round', 'is_second_round', 'day_Fri',
                    'day_Mon', 'day_Sat', 'day_Sun', 'day_Thu', 'day_Tue', 'day_Wed',
                    'venue_Away', 'venue_Home', 'home_team_encoded', 'opponent_team_encoded', 
                    'formation_defenders', 'formation_midfielders', 'formation_strikers']
        
        cols = ['date', 'team', 'opponent', 'season'] + cont_cols + cat_cols
        
        new_data = new_data[cols]
        new_data['date'] = pd.to_datetime(new_data['date'])

        return new_data
    
