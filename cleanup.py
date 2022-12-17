import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from scipy.sparse import csr_matrix

# warnings are distracting...
import warnings
warnings.filterwarnings("ignore")

# read raw data
NBA_Shot_Logs = pd.read_csv('./data/shot_logs.csv')

# drop irrelevant features
NBA_Shot_Logs = NBA_Shot_Logs.drop(['GAME_ID', 'player_name','FGM','MATCHUP','CLOSEST_DEFENDER','PTS'], axis=1)

# enable for data without categorical variables
#NBA_Shot_Logs = NBA_Shot_Logs.drop(['CLOSEST_DEFENDER_PLAYER_ID', 'player_id'], axis=1)

# change 'GAME_CLOCK' to second representation
NBA_Shot_Logs['GAME_CLOCK'] = NBA_Shot_Logs['GAME_CLOCK'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

# drop na data (there's no shot clock if game clock < 24s)
NBA_Shot_Logs = NBA_Shot_Logs.dropna(how = 'any', axis = 0)

# convert alphabets/words to numbers
NBA_Shot_Logs['LOCATION'][NBA_Shot_Logs.LOCATION == 'H'] = 1
NBA_Shot_Logs['LOCATION'][NBA_Shot_Logs.LOCATION == 'A'] = 0
NBA_Shot_Logs['W'][NBA_Shot_Logs.W == 'W'] = 1
NBA_Shot_Logs['W'][NBA_Shot_Logs.W == 'L'] = 0
NBA_Shot_Logs['SHOT_RESULT'][NBA_Shot_Logs.SHOT_RESULT == 'made'] = 1
NBA_Shot_Logs['SHOT_RESULT'][NBA_Shot_Logs.SHOT_RESULT == 'missed'] = 0

# convert ids from numbers to strings
NBA_Shot_Logs['CLOSEST_DEFENDER_PLAYER_ID'] = NBA_Shot_Logs['CLOSEST_DEFENDER_PLAYER_ID'].astype(str)

# enable for one-hot encoding on categorical variables
#transformer = make_column_transformer(
#    (OneHotEncoder(), ['CLOSEST_DEFENDER_PLAYER_ID', 'player_id']), remainder='passthrough')
#transformed = transformer.fit_transform(NBA_Shot_Logs)

# the matrix is transformed into sparse representation since it's too large...
#transformed_dense = csr_matrix.toarray(transformed)
#transformed_df = pd.DataFrame(transformed_dense, columns=transformer.get_feature_names())
#transformed_df.to_csv('onehotencoding_shot_logs.csv')

# check if drop all na
#print(NBA_Shot_Logs.isnull().any())

# Display result data
#NBA_Shot_Logs.to_csv('clean.csv')
print(NBA_Shot_Logs.head())
