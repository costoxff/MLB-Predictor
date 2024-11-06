import pandas as pd

class MLBParser:
    """MLB dataset parser

    Attributes
    ----------
    __raw_df : pandas.DataFrame
        Original dataframe
    dataframe : pandas.DataFrame
        A DataFrame return from function self.__preprocessing
    """
    def __init__(self, filename: str, preprocessing=None):
        self.__raw_df = pd.read_csv(filename, index_col=0)
        if preprocessing is not None:
            self.dataframe = preprocessing(self.__raw_df)
        else:
            self.dataframe = self.__preprocessing(self.__raw_df)
    
    def __preprocessing(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        naive MLB dataset pre-processing

        :param raw_df: MLB dataframe readed by pandas.read_csv()
        """
        df = raw_df.copy()

        # remove unused columns
        unused = ['date', 'season', 'home_team_season', 'away_team_season',
                  'home_team_rest', 'away_team_rest', 'home_pitcher_rest', 
                  'away_pitcher_rest']
        df.drop(unused, axis=1, inplace=True)

        df['is_night_game'] += 0 # bool -> int
        df['home_team_win'] += 0 # bool -> int

        # team abbreviation maping: str -> int
        team_set = set(df['home_team_abbr'].tolist() + df["away_team_abbr"].tolist())
        team_dict = {list(team_set)[i]: i for i in range(len(team_set))}
        df['home_team_abbr'] = df['home_team_abbr'].map(lambda x : team_dict[x])
        df['away_team_abbr'] = df['away_team_abbr'].map(lambda x : team_dict[x])

        # pithcer maping: str -> int
        pitcher_set = set(df['home_pitcher'].tolist() + df["away_pitcher"].tolist())
        pitcher_dict = {list(pitcher_set)[i]: i for i in range(len(pitcher_set))}
        df['home_pitcher'] = df['home_pitcher'].map(lambda x : pitcher_dict[x])
        df['away_pitcher'] = df['away_pitcher'].map(lambda x : pitcher_dict[x])

        df.dropna(inplace=True) # remove rows with "Nan"
        return df

    def get_raw_df(self):
        return self.__raw_df

    def get_game_info(self):
        return self.__raw_df[[
            'home_team_abbr',
            'away_team_abbr',
            'date',
            'is_night_game',
            'home_team_win',
            'season'
        ]]
    
    def get_team_pitcher_rest(self, hosts=['home']):
        """
        :param hosts: A list contain 'home' or 'away'
        """
        columns = []
        for h in hosts:
            columns += [
                f'{h}_team_rest',
                f'{h}_pitcher_rest',
            ]
        return self.__raw_df[columns]
    
    def get_pitcher_info(self, hosts=['home']):
        """
        :param hosts: A list contain 'home' or 'away'
        """
        columns = []
        for h in hosts:
            columns += [f'{h}_pitcher']
        return self.__raw_df[columns]
    
    def get_recent_perform(self, hosts=['home']):
        """
        :param hosts: A list contain 'home' or 'away'
        """
        columns = []
        for h in hosts:
            columns += [
                f'{h}_batting_batting_avg_10RA',
                f'{h}_batting_onbase_perc_10RA',
                f'{h}_batting_onbase_plus_slugging_10RA',
                f'{h}_batting_leverage_index_avg_10RA',
                f'{h}_batting_RBI_10RA',
                f'{h}_pitching_earned_run_avg_10RA',
                f'{h}_pitching_SO_batters_faced_10RA',
                f'{h}_pitching_H_batters_faced_10RA',
                f'{h}_pitching_BB_batters_faced_10RA',
                f'{h}_pitcher_earned_run_avg_10RA',
                f'{h}_pitcher_SO_batters_faced_10RA',
                f'{h}_pitcher_H_batters_faced_10RA',
                f'{h}_pitcher_BB_batters_faced_10RA',
            ]
        return self.__raw_df[columns]
    
    def get_team_seasonal_statistic(self, hosts=['home'], statis=['mean']):
        """
        :param hosts: A list contain 'home' or 'away'
        :param statis: A list contain 'mean' or 'std' or 'skew'
        """
        columns = []
        for h in hosts:
            columns += [f'{h}_team_season']
            for sta in statis:
                columns += [
                    f'{h}_team_errors_{sta}',
                    f'{h}_team_spread_{sta}',
                    f'{h}_team_wins_{sta}',
                ]
        return self.__raw_df[columns]
    
    def get_seasonal_battle(self, hosts=['home'], statis=['mean']):
        """
        :param hosts: A list contain 'home' or 'away'
        :param statis: A list contain 'mean' or 'std' or 'skew'
        """
        columns = []
        for h in hosts:
            for sta in statis:
                columns += [
                    f'{h}_batting_batting_avg_{sta}',
                    f'{h}_batting_onbase_perc_{sta}',
                    f'{h}_batting_onbase_plus_slugging_{sta}',
                    f'{h}_batting_leverage_index_avg_{sta}',
                    f'{h}_batting_wpa_bat_{sta}',
                    f'{h}_batting_RBI_{sta}',
                ]
        return self.__raw_df[columns]

    def get_team_seasonal_pitching(self, hosts=['home'], statis=['mean']):
        """
        :param hosts: A list contain 'home' or 'away'
        :param statis: A list contain 'mean' or 'std' or 'skew'
        """
        columns = []
        for h in hosts:
            for sta in statis:
                columns += [
                    f'{h}_pitching_earned_run_avg_{sta}',
                    f'{h}_pitching_SO_batters_faced_{sta}',
                    f'{h}_pitching_H_batters_faced_{sta}',
                    f'{h}_pitching_BB_batters_faced_{sta}',
                    f'{h}_pitching_leverage_index_avg_{sta}',
                    f'{h}_pitching_wpa_def_{sta}',
                ]
        return self.__raw_df[columns]

    def get_pitcher_seasonal_perform(self, hosts=['home'], statis=['mean']):
        """
        :param hosts: A list contain 'home' or 'away'
        :param statis: A list contain 'mean' or 'std' or 'skew'
        """
        columns = []
        for h in hosts:
            for sta in statis:
                columns += [
                    f'{h}_pitcher_earned_run_avg_{sta}',
                    f'{h}_pitcher_SO_batters_faced_{sta}',
                    f'{h}_pitcher_H_batters_faced_{sta}',
                    f'{h}_pitcher_BB_batters_faced_{sta}',
                    f'{h}_pitcher_leverage_index_avg_{sta}',
                    f'{h}_pitcher_wpa_def_{sta}',
                ]
        return self.__raw_df[columns]