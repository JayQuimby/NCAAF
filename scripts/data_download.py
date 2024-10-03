from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from bs4 import BeautifulSoup
from datetime import date
import pandas as pd
import numpy as np
import requests
import pickle
import time
import re
import tensorflow as tf
label_encoder = LabelEncoder()

def read_teams():
  read_list = []
  # Open the file in read mode
  with open('./scripts/team_names.txt', 'r') as file:
      for line in file:
          # Append each line (after stripping newline characters) to the list
          read_list.append(line.strip())       
  return read_list

labels = ['Games', 'pass_completion', 'pass_attempts', 'pass_completion_percent', 'pass_yards', 'pass_td', 'rush_attempt', 'rush_yards', 'avg_rush_yard', 'rush_td', 'num_plays', 'total_yard', 'yard_per_play', 'pass_first_down', 'rush_first_down', 'penalty_first', 'first_downs', 'penalty', 'penalty_yards', 'lost_fumble', 'interceptions', 'turnovers']
op_labels = ['op_'+s for s in labels]
yearly_stats_cols = ['Year','Conf','W_overall','L_overall','T_overall','Pct_overall','W_conf','L_conf','T_conf','Pct_conf','SRS','SOS','AP_Pre','AP_High','AP_Post','CFP_High','CFP_Final','Coach','Bowl','Notes']
hist_cols = ['Date', 'Location', 'Opponent', 'Result', 'pass_completion', 'pass_attempts', 'pass_completion_percent', 'pass_yards', 'pass_td', 'rush_attempt', 'rush_yards', 'avg_rush_yard', 'rush_td', 'num_plays', 'total_yard', 'yard_per_play', 'pass_first_down', 'rush_first_down', 'penalty_first', 'first_downs', 'penalty', 'penalty_yards', 'lost_fumble', 'interceptions', 'turnovers'] 
#offense_cols = hist_cols[:4] + ['of_' + stat for stat in hist_cols[4:]]
#defense_cols = ['de_' + stat for stat in hist_cols[4:]]
#all_cols = hist_cols + defense_cols
#df_cols = all_cols[4:] + ['op_' + s for s in all_cols[4:]]
this_year = 2023
teams = read_teams()
schedule_url_base = 'https://www.sports-reference.com/cfb/schools/%s/%i-schedule.html'
stats_url_base = 'https://www.sports-reference.com/cfb/schools/%s/%i.html'
stats_hist_url_base = 'https://www.sports-reference.com/cfb/schools/%s/%i/gamelog/'
yearly_stats_url_base = 'https://www.sports-reference.com/cfb/schools/%s/'


def save_data(data: object, filename: str) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(filename: str) -> object:
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def amortize(table):
    rolling_avg_table = table.copy()
    rolling_avg_table[hist_cols[4:]] = rolling_avg_table[hist_cols[4:]].apply(pd.to_numeric)
    #print(rolling_avg_table)
    for idx in range(1, table.shape[0]):
        rolling_avg = table.iloc[:idx, 4:].mean()
        rolling_avg_table.iloc[idx, 4:] = rolling_avg
    rolling_avg_table = rolling_avg_table.iloc[1:]
    #print(rolling_avg_table)
    return rolling_avg_table

def get_team_stats(url: str) -> pd.DataFrame:
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  table = soup.find('tbody')
  data = []

  for row in table.find_all('tr'):
    cells = row.find_all('td')
    #print(cells[0])
    if len(cells) > 0:
      row_data = [cell.text.strip() for cell in cells]
      #print(row_data)
      data.append(row_data)

  df = pd.DataFrame(data)
  return df

def get_yearly_stats(url, team, yearly_cols = yearly_stats_cols):
  data = []
  response = requests.get(url % team)
  soup = BeautifulSoup(response.text, 'html.parser')
  table = soup.find('tbody')
  if table:
    for row in table.find_all('tr'):
      cols = row.find_all('td')
      if len(cols) > 0:
        row_data = [col.text.strip() for col in cols]
        #print(row_data)
        data.append(row_data)
  df = pd.DataFrame(data)
  df.columns = yearly_cols
  return df

def get_team_stats_hist(url: str) -> pd.DataFrame:
  data = []
  response_1 = requests.get(url + '#offense')
  response_2 = requests.get(url + '#defense')
  soups = [BeautifulSoup(response_1.text, 'html.parser'), BeautifulSoup(response_2.text, 'html.parser')]
  for soup in soups:
    table = soup.find('tbody')
    if table:
      for row in table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) > 0:
          row_data = [cell.text.strip() for cell in cells]
          data.append(row_data)
  df = pd.DataFrame(data)
  top = df.iloc[:int(df.shape[0]/2), :]
  bot = df.iloc[int(df.shape[0]/2):, :]
  bot = bot.iloc[:,4:]
  bot.reset_index(drop=True, inplace=True)
  new_df = pd.concat([top, bot], axis=1)
  new_df.columns = all_cols
  new_df.iloc[:, 4:] = new_df.iloc[:,4:].map(pd.to_numeric)
  new_df = amortize(new_df)
  return new_df

def get_team_schedule(url: str) -> pd.DataFrame:
  labels_new = ['Date', 'Time', 'Day', 'Team', 'Location', 'Opponent', 'Conference', 'W/L', 'Points', 'Op_points', 'Wins', 'Losses', 'Streak', 'Notes']
  labels_old = labels_new[0:1] + labels_new[2:]
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  table = soup.find_all('table')
  table = table[1] if len(table) > 1 else table[0]
  data = []
  for row in table.find_all('tr'):
    cells = row.find_all('td')
    
    if len(cells) > 0:
      row_data = [cell.text.strip() for cell in cells]
      data.append(row_data)

  df = pd.DataFrame(data)
  df.columns = labels_new if len(df.columns) == 14 else labels_old
  return df

oddballs = load_data('./name_conversion.pkl')

def standardize(input: str) -> str:
  input = re.sub(r'\(\d*\)', '', input)
  input = re.sub(r'\xa0', '', input)
  input = input.replace(' ', '-').replace('(', '').replace(')', '').replace('*', '').lower()
  if input in oddballs:
    input = oddballs[input]
  return input

def name_standard(input: str):
   return re.sub(r'\s*\(\d+-\d+\)$', '', input)

from datetime import datetime

def convert_date(date_str):
    try:
        # Attempt to parse the input date using both formats
        date_formats = ['%b %d, %Y', '%Y-%m-%d']
        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, date_format)
                # Convert the parsed date back to the second format
                formatted_date = parsed_date.strftime('%Y-%-m-%-d')
                return formatted_date
            except ValueError:
                continue

        # If neither format matches, raise an exception
        raise ValueError("Invalid date format")

    except Exception as e:
        return str(e)

def get_cur_teams() -> list:    
    teams = get_team_stats('https://www.sports-reference.com/cfb/schools/')
    teams = teams[teams[2] == '2023'][0]
    #teams = teams[teams[1] <= '2010'][0]
    teams = [standardize(s) for s in teams]
    return teams


selector = ['Year','SRS','SOS']#,'AP_Pre','AP_High','Coach']
def get_all_yearly_stats(teams_names, use_old=False):
    team_yearly_data = {} if not use_old else load_data('./Data/yearly/SRS.pkl')

    for team in teams_names:
        if team in team_yearly_data.keys():
            continue

        dataframe = get_yearly_stats(yearly_stats_url_base, team)
        dataframe = dataframe[dataframe['Year'] > '2009']
        #dataframe['Coach'] = dataframe['Coach'].apply(name_standard)
        dataframe['Year'] = dataframe['Year'].astype(int)
        dataframe[['SRS', 'SOS']] = dataframe[['SRS', 'SOS']].replace('', 0.0).astype(float)
        dataframe = dataframe[selector]
        team_yearly_data[team] = dataframe
        time.sleep(3.2)
    return team_yearly_data

def get_all_stats_year(year):
    stats_dict = {}
        # Loop through teams to extract stats, odds, schedule
    for team in teams:
        if team not in stats_dict.keys():
            #print(team)
            all_data = []
            team_link = stats_url_base % (team, year)
            try:
                stats = get_team_stats(team_link)
                all_data.append(stats)
                df = pd.concat(all_data)
                df.columns = ['Games', 'pass_completion', 'pass_attempts', 'pass_completion_percent', 'pass_yards', 'pass_td', 'rush_attempt', 'rush_yards', 'avg_rush_yard', 'rush_td', 'num_plays', 'total_yard', 'yard_per_play', 'pass_first_down', 'rush_first_down', 'penalty_first', 'first_downs', 'penalty', 'penalty_yards', 'lost_fumble', 'interceptions', 'turnovers']  # IMPORTANT -------------- !!!
                stats_dict[team] = df
                time.sleep(3.2)
            except Exception as e:
                print(f'Error occured: {e}')
                continue
                #play_error_sound()
    return stats_dict

def get_all_schedules_year(year, verbose=False):
    s_dict = {}
    for team in teams:
        if team not in s_dict.keys():
            team_link = schedule_url_base % (team, year)
            try:
                if verbose:
                    print(f'Getting {team} schedule for {year}')
                schedule = get_team_schedule(team_link)
                s_dict[team] = schedule
                time.sleep(3.2)
            except Exception as e:
                if verbose:
                    print(f'Error occured: {e}')
                time.sleep(3.2)
                continue
                #play_error_sound()
    return s_dict

def get_all_stats_hist_year(year, verbose=False):
    stats_dict = {}
    # Loop through teams to extract stats, odds, schedule
    for team in teams:
        if team not in stats_dict.keys():
            print(team)
            team_link = stats_hist_url_base % (team, year)
            try:
                stats = get_team_stats_hist(team_link)
                df = pd.DataFrame(stats)
                df.columns = all_cols  # IMPORTANT -------------- !!!
                #print(df.shape)
                stats_dict[team] = df
                time.sleep(6.2)
            except Exception as e:
                if verbose:
                    print(f'Error occured: {e}')
                time.sleep(3.2)
                continue
    return stats_dict

def download_all_stats(making_predictions=True):
    for x in range(2010,2024) if not making_predictions else range(2023,2024):
        print(f'Starting year {x}')
        stats_ = get_all_stats_year(x)
        save_data(stats_, f'./Data/Stats/{x}_stats.pkl')

def download_all_schedules(making_predictions=True):
    for x in range(2010,2024) if not making_predictions else range(2023,2024):
        print(f'Starting year {x}')
        games_ = get_all_schedules_year(x)
        save_data(games_, f'./Data/Schedules/{x}_schedule.pkl')

def download_all_stats_hist():
    for x in range(2010,2024):
        print(f'Starting year {x}')
        stats_ = get_all_stats_hist_year(x, True)
        save_data(stats_, f'./Data/Stats/{x}_history.pkl')

def stand_key(di):
    for year in di.values():
        for name in year.keys():
            name = standardize(name)
    return di

def load_stats_data(filepath = './Data/Stats/%i_stats.pkl', making_predictions=True):
    data_dict = {}
    for yr in range(2010, 2024) if not making_predictions else range(this_year, this_year+1):
        yr_data = load_data(filepath % yr)
        data_dict[yr] = yr_data
    data_dict = stand_key(data_dict)
    return data_dict

def make_cur_dataframe(stats, schedule, all_season=False):
    filter_cols = ['Date','Team','Location','Opponent','Conference','Points','W/L']
    data_set = []
    for team, schedule in schedule.items():

        if 'Time' in schedule.columns:
            schedule.drop('Time', axis=1, inplace=True)

        schedule['Date'] = schedule['Date'].apply(convert_date)
        team_stats = stats[team].iloc[0].to_list()[1:]
        schedule[['Team','Opponent']] = schedule[['Team','Opponent']].map(standardize)
        if not all_season:
            schedule = schedule[schedule['W/L'] == '']
            schedule = schedule.reset_index()
        
        for i, row in schedule.iterrows():
            try:
                opp_stats = stats[row['Opponent']].iloc[1].to_list()[1:]
                
            except Exception as e:
                print(e)
                continue
            categorical = row[filter_cols].to_list()
            example = categorical + team_stats + opp_stats
            data_set.append(example)
 
    data_df = pd.DataFrame(data_set)
    data_df.columns = filter_cols[:-2] + ['Label','Binary_Label'] + labels[1:] + ['SRS','SOS'] + op_labels[1:] + ['op_SRS','op_SOS']
    teams = data_df[['Team', 'Opponent']]
    value_mapping = load_data('./Data/Schedules/value_map.pkl')
    #print(value_mapping)
    data_df['Team'] = data_df['Team'].map(value_mapping).astype('float32')
    data_df['Opponent'] = data_df['Opponent'].map(value_mapping).astype('float32')
    data_df[['Binary_Label']] = data_df[['Binary_Label']].apply(label_encoder.fit_transform).astype('int16') 
    data_df[['Location','Conference']] = data_df[['Location','Conference']].apply(label_encoder.fit_transform).astype('int32')
    data_df[['Team_name', 'Op_name']] = teams
    return data_df

def make_dataframe(stats, schedules, years=range(2013,this_year)):
    filter_cols = ['Team','Location','Opponent','Conference','Points','W/L']
    data_set = []
    
    for year in years:
        print(year, len(data_set), sep=':')
        for team, schedule in schedules[year].items():

            if 'Time' in schedule.columns:
                schedule.drop('Time', axis=1, inplace=True)

            schedule['Date'] = schedule['Date'].apply(convert_date)
            cur_stats = stats[year][team]
            #print(cur_stats)
            
            schedule[['Team','Opponent']] = schedule[['Team','Opponent']].map(standardize)
            schedule = schedule[schedule['W/L'] != '']
            schedule = schedule.reset_index()

            for i, row in schedule.iterrows():
                if i == 0:
                    continue

                #team_stats = cur_stats[cur_stats['Date'] == row['Date']].squeeze()
                team_stats = cur_stats.iloc[0]
                if team_stats.empty:
                    continue
                team_stats = team_stats.tolist()[1:]
                
                try:
                    opp_stats = stats[year][row['Opponent']]
                    #opp_stats['Date'] = opp_stats['Date'].apply(convert_date)
                    #opp_stats = opp_stats[opp_stats['Date'] == row['Date']].squeeze()
                    opp_stats = opp_stats.iloc[1]
                    if opp_stats.empty or type(opp_stats) != pd.core.series.Series:
                        continue
                    opp_stats = opp_stats.tolist()[1:]
                    #print('opp', opp_stats)
                    
                except Exception as e:
                    continue
                
                
                categorical = row[filter_cols].to_list()
                example = categorical + team_stats + opp_stats 
                data_set.append(example)
 
    data_df = pd.DataFrame(data_set)
    data_df.columns = filter_cols[:-2] + ['Label','Binary_Label'] + labels[1:] + ['SRS','SOS'] + op_labels[1:] + ['op_SRS','op_SOS']
    value_mapping = load_data('./Data/Schedules/value_map.pkl')
    data_df['Team'] = data_df['Team'].map(value_mapping).astype('float32')
    data_df['Opponent'] = data_df['Opponent'].map(value_mapping).astype('float32')
    data_df[['Binary_Label']] = data_df[['Binary_Label']].apply(label_encoder.fit_transform).astype('int16') 
    data_df[['Location','Conference']] = data_df[['Location','Conference']].apply(label_encoder.fit_transform).astype('int32')
    data_df = obj_to_float(data_df)
    return data_df

def obj_to_float(data):
    object_columns = data.select_dtypes(include=['object']).columns
    data[object_columns] = data[object_columns].astype('float32')
    return data

def normalize_data(data, scaler_save='./Dataset/MMScaler_.pkl', use_save=False):
  y = data[['Label', 'Binary_Label']] 
  X = data.drop(['Label', 'Binary_Label'] , axis=1)
  scaler = MinMaxScaler() if not use_save else load_data(scaler_save)
  scaler.fit(X)
  if not use_save:
    pickle.dump(scaler, open(scaler_save, 'wb')) 
  X_scaled = scaler.transform(X)
  X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
  df_scaled = pd.concat([y, X_scaled], axis=1)
  return df_scaled, scaler

def add_srs(stats_data, year=this_year):
    #if 'SRS' in stats_data['georgia'].columns:
    #    return stats_data
    yearly_data = load_data('./Data/yearly/SRS.pkl')
    for team, vals in stats_data.items():
        team_add = yearly_data[team]
        team_row = team_add[team_add['Year'] == year]
        team_row = team_row.drop('Year', axis = 1)
        team_row = tf.concat([team_row] * 3, axis=0)
        vals[['SRS', 'SOS']] = team_row
    return stats_data

def add_all_srs(stats_data):
    for year in stats_data.keys():
        if year < 2013:
            continue
        stats_data[year] = add_srs(stats_data[year], year)
    return stats_data

def download_and_update_datasets():
    download_all_schedules(True)
    download_all_stats(True)
    new_srs = get_all_yearly_stats(teams)
    save_data(new_srs, './Data/yearly/SRS.pkl')
    update_this_year()
    update_dataframes('_srs')
    
def update_this_year():
    data_update = load_data(f'./Data/Stats/{this_year}_stats.pkl')
    data_update = add_srs(data_update)
    save_data(data_update, f'./Data/Stats/{this_year}_stats.pkl')

def update_dataframes(addon):
    stats_data_dict = load_stats_data(making_predictions=False)
    schedule_data_dict = load_stats_data(filepath = './Data/Schedules/%i_schedule.pkl', making_predictions=False)
    stats_data_dict = add_all_srs(stats_data_dict)
    print(stats_data_dict[2023]['georgia'].columns)
    print(stats_data_dict[2018]['georgia'].columns)
    full_dataset = make_dataframe(stats_data_dict, schedule_data_dict)
    object_columns = full_dataset.select_dtypes(include=['object']).columns
    full_dataset[object_columns] = full_dataset[object_columns].astype('float32')
    new_data, _ = normalize_data(full_dataset, f'./Dataset/MMScaler{addon}.pkl', True)
    save_data(new_data, f'./Dataset/scaled_dataset{addon}.pkl')

def get_week_inputs(week=None, all_season=False, update=False):
    if update:
        download_and_update_datasets()
    stats_ = load_stats_data(making_predictions=True)
    sched_ = load_stats_data('./Data/Schedules/%i_schedule.pkl', True)
    
    if all_season:
        future = make_cur_dataframe(stats_[this_year], sched_[this_year], True)
        save_data(future, f'./Data/predictions/all_season.pkl')
    else:
        cur_frame = make_cur_dataframe(stats_[this_year], sched_[this_year])
        this_weekend = cur_frame[cur_frame['Date'].apply(standardize).isin(week)]
        #print(this_weekend['Date'])
        this_weekend = this_weekend.reset_index(drop=True)
        save_data(this_weekend, f'./Data/predictions/{date.today()}.pkl')

def make_stats_data_table():
    cur_stats = load_data(filename=f'Data/Stats/{this_year}_stats.pkl')
    res = pd.DataFrame()
    for team, df in cur_stats.items():
        df = df.drop(2)
        cols = df.columns
        offense = df.iloc[0]
        defense = df.iloc[1][1:-2]
        concat_df = pd.concat([offense, defense])
        concat_df = pd.DataFrame(concat_df).T
        concat_df.columns = list(cols) + ['de_' + col for col in cols[1:-2]]
        concat_df['team_name'] = team
        res = pd.concat([res, concat_df], ignore_index=True)
    return res

