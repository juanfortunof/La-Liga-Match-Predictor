import pandas as pd
import numpy as np
import pp_train as c


def form_constr(data: pd.DataFrame, team_col: str, formation_col: str) -> dict:

    if not isinstance(data, pd.DataFrame): raise ValueError('Data must be a pandas Dataframe')

    if not isinstance(team_col, str) or \
        not isinstance(formation_col, str):\
            raise ValueError('team_col and formation_col must be strings')
       

    '''
        This function builds the test data defenders, midfielders and strikers
        columns using the most repeated formations of the team in the past years
        and placed them into a dictionary.
    '''

    grouped_teams = data.groupby(team_col)[formation_col].value_counts().reset_index()
    team_indexes_max = grouped_teams.groupby(team_col)['count'].idxmax()

    teams_dict = grouped_teams.loc[team_indexes_max, [team_col, formation_col]].to_dict()
    teams_players_dict = {team: int(players) for team, players in zip(teams_dict[team_col].values(), teams_dict[formation_col].values())}

    return teams_players_dict


def reuse_functionality(data: pd.DataFrame, date_col: str, round_col: str, one_hot_cols: str,
                        home_team_col: str, opp_team_col: str):

    new_data = data.copy()

    month_obj = c.month_transf(data, date_col)
    new_data = month_obj.first_sec_part_and_month(new_data)

    round_obj = c.round_transf(data, round_col)
    new_data = round_obj.encode_col(new_data)
    new_data = round_obj.is_first_round(new_data)
    new_data = round_obj.is_second_round(new_data)

    one_hot_obj = c.one_hot_encoder(data, one_hot_cols)
    new_data = one_hot_obj.encode(new_data)

    col_enc_obj = c.column_encoder(data, 'a', home_team_col, opp_team_col)
    new_data = col_enc_obj.encode_team(new_data)

    return new_data


# ARREGLAR ESTO MAÑANA, EL TEST DATASET LE FALTAN ALGUNAS COLUMNAS DEL FEATURE ENGINEERING, QUIZAS UNA SOLUCION PUEDA SER 
# HACERLE UNPACKING A UNA LISTA ORDENADA.


def cont_hist_avgs(training_data: pd.DataFrame, new_data: pd.DataFrame, team_col: str,
                   xg_col: str, xga_col: str, poss_col: str, gf_col: str, 
                   ga_col: str, sot_col: str, sh_col: str, is_test_set:bool):

    cont_cols_obj = c.feature_engineer(training_data, xg_col, xga_col, poss_col, gf_col, ga_col, sot_col, sh_col)
    mod_training_data, cont_cols = cont_cols_obj.new_eng_cols(training_data)
    if not is_test_set:
        new_data = new_data.drop(columns=[xg_col, xga_col, poss_col, gf_col, ga_col, sot_col], axis=1)

    averages = mod_training_data.groupby(team_col)[cont_cols].mean()
    
    least_5_teams = mod_training_data[team_col].value_counts().tail().index.tolist()
    least_5 = mod_training_data[mod_training_data[team_col].isin(least_5_teams)]
    avgs_for_Oviedo = least_5.groupby(team_col)[cont_cols].mean().mean()
    averages.loc['Oviedo'] = avgs_for_Oviedo

    for col in averages.columns:
        new_data[col] = new_data[team_col].map(averages[col])

    return new_data


def merge_all(training_data: pd.DataFrame, new_data: pd.DataFrame, date_col: str, round_col: str, one_hot_cols: str,
              home_team_col: str, opp_team_col: str, formation_col: str,
              def_col: str, mid_col: str, str_col: str, xg_col: str, xga_col: str, poss_col: str, gf_col: str, ga_col: str, 
              sot_col: str, sh_col: str, result_col: str, test_set: bool):

    new_data = reuse_functionality(new_data, date_col, round_col, one_hot_cols, 
                                    home_team_col, opp_team_col)
    
    new_data = cont_hist_avgs(training_data, new_data, home_team_col, xg_col, 
                              xga_col, poss_col, gf_col, ga_col, sot_col, sh_col, test_set)
    
    if not test_set:
        form_obj = c.form_clean_transf(training_data, formation_col)
        training_data = form_obj.separate_formation(training_data)

        defenders_map = form_constr(training_data, home_team_col, def_col)
        defenders_map['Oviedo'] = 4
        midfielders_map = form_constr(training_data, home_team_col, mid_col)
        midfielders_map['Oviedo'] = 4
        strikers_map = form_constr(training_data, home_team_col, str_col)
        strikers_map['Oviedo'] = 2

        new_data['formation_defenders'] = new_data[home_team_col].map(defenders_map)
        new_data['formation_midfielders'] = new_data[home_team_col].map(midfielders_map)
        new_data['formation_strikers'] = new_data[home_team_col].map(strikers_map)

    col_enc_obj = c.column_encoder(new_data, result_col, home_team_col, opp_team_col)
    new_data = col_enc_obj.encode_target(new_data)

    cat_cols = ['result', 'month', 'played_first_part_of_month', 'played_second_part_of_month',
                'round_encoded', 'is_first_round', 'is_second_round', 'day_Fri', 
                'day_Mon', 'day_Sat', 'day_Sun', 'day_Thu', 'day_Tue', 'day_Wed',
                'venue_Away', 'venue_Home', 'home_team_encoded', 'opponent_team_encoded', 
                'formation_defenders', 'formation_midfielders', 'formation_strikers']
    
    cont_cols = ['gf', 'ga', 'xg', 'xga', 'poss', 'sot', 'sh', 'dist', 'fk', 'pk', 'pkatt',
                 'xg_diff', 'poss_a', 'poss_diff', 'effective_possesion', 'vulnerable_possesion', 
                 'gf_ga_diff', 'shooting_scoring_efficiency', 'shooting_efficiency']
    
    if test_set:
        cat_cols.remove('formation_defenders')
        cat_cols.remove('formation_midfielders')
        cat_cols.remove('formation_strikers')
    
    cols = ['date', 'team', 'opponent', 'season'] + cont_cols + cat_cols
    new_data['date'] = pd.to_datetime(new_data['date'])

    if test_set:
        new_data['result'] = np.nan

    new_data = new_data[new_data['venue_Home'] == 1]
    new_data = new_data.sort_values(by='date')
    
    return new_data[cols]
