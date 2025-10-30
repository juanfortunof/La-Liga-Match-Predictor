import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_rolling_averages(data: pd.DataFrame, new_cols: list, window_size: int, date_col) -> pd.DataFrame:

    '''
    This is what this function does:

    First sort the dataframe by the date, so the rolling averages can be
    calculated correctly, then using the cols selected, the function calculates
    the rolling averages for each and they add them to the original dataframe using
    the new_cols name. In order to use this
    function you have to group by the team column and apply the function this way:

    data.groupby(team_column).apply(lambda: x get_rolling_averages(x, cols, new_cols)
    '''

    if not isinstance(data, pd.DataFrame): raise ValueError('Data must be a pandas DataFrame')

    if not isinstance(new_cols, list): raise ValueError('cols argument must be a list of strings')

    if not new_cols: raise ValueError('cols list must have something')

    if window_size < 1: raise ValueError('Window size must be positive and at least 1')

    if not pd.api.types.is_datetime64_any_dtype(data[date_col].dtype): raise ValueError('The date column should be of datetime type')

    data = data.copy()

    data = data.sort_values(date_col)

    try:
        
        rolling_stats = data[new_cols].rolling(window=window_size, closed='left').mean()
    
        new_rolling_cols = [f'rolling_{col}' for col in new_cols]
        data[new_rolling_cols] = rolling_stats
        data = data.dropna(subset=new_rolling_cols)
        data = data.drop(new_cols, axis=1)
        rename_map = {'rolling_' + col: col for col in new_cols}
        data = data.rename(rename_map, axis=1)
        
    except Exception as e:
        
        logger.error(f'Error calculating rolling averages {str(e)}')
        raise

    return data


def get_historical_by_season(data: pd.DataFrame, cols_to_calc: list, team_col: str, season_col: str) -> pd.DataFrame:

    '''
        This function is pretty straight forward, we're just going to calculate
        the historic averages for every team by each season.
    '''

    if not isinstance(data, pd.DataFrame): raise ValueError('Data must be a pandas DataFrame')

    if not isinstance(cols_to_calc, list) or not cols_to_calc: raise ValueError('cols argument must be a list of strings')

    if not isinstance(team_col, str): raise ValueError('Team column must be a string')

    if not isinstance(season_col, str): raise ValueError('Season column must be a string')

    cols_to_merge = [team_col, season_col]

    historic_data = data.groupby(cols_to_merge)[cols_to_calc].mean().reset_index()
    historic_avgs = data.merge(historic_data, on=cols_to_merge, how='left', suffixes=('', '_historic_avg'))
    historic_avgs = historic_avgs.drop(cols_to_calc, axis=1)

    rename_mapper = {col + '_historic_avg': col for col in cols_to_calc}

    historic_avgs = historic_avgs.rename(rename_mapper, axis=1)
    
    return historic_avgs