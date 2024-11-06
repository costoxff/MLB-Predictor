from datasetutils import *

def lambda_func(x):
    print("inner function")
    return 'data return from function'

# constructor testing
mlb = MLBParser("./stage-1/train_data.csv", preprocessing=lambda_func)
assert mlb is not None

assert mlb.get_game_info() is not None

hora = [['home'], ['away'], ['home', 'away']]
statiss = [
    ['mean'],
    ['std'],
    ['skew'],
    ['mean', 'std'],
    ['mean', 'skew'],
    ['std', 'skew'],
    ['mean', 'std', 'skew']
]

# object method testing
for i in hora:
    assert mlb.get_team_pitcher_rest(i) is not None
    assert mlb.get_pitcher_info(i) is not None
    assert mlb.get_recent_perform(i) is not None
    for j in statiss:
        assert mlb.get_team_seasonal_statistic(i, j) is not None
        assert mlb.get_seasonal_battle(i, j) is not None
        assert mlb.get_team_seasonal_pitching(i, j) is not None
        assert mlb.get_pitcher_seasonal_perform(i, j) is not None