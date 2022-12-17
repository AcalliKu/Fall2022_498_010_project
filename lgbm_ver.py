import lightgbm
from lightgbm import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score

print('reading data...')
NBA_Shot_Logs = pd.read_csv('./shot_log_clean.csv')
print('finish reading data...')

#NBA_Shot_Logs = pd.DataFrame(NBA_Shot_Logs, columns=NBA_Shot_Logs.get_feature_names())

features = NBA_Shot_Logs.columns.values.tolist()
features.pop(0)
features.remove('SHOT_RESULT')

print(features)

for feature in ['CLOSEST_DEFENDER_PLAYER_ID', 'player_id', 'PERIOD']:
    NBA_Shot_Logs[feature] = pd.Series(NBA_Shot_Logs[feature], dtype="category")

data_X_df = NBA_Shot_Logs[features]
data_Y_df = NBA_Shot_Logs['SHOT_RESULT']

data_X = data_X_df.values.tolist()
data_Y = data_Y_df.values.tolist()

sc = StandardScaler()
sc.fit(data_X)
data_X = sc.transform(data_X)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['l2', 'auc'],
    'learning_rate': 0.005,
    "num_leaves": 128,
    "max_bin": 512,
}


lgb_train = lightgbm.Dateset(X_train, y_train)
gbm = lightgbm.train(hyper_params, lgb_train, num_boost_round=10, verbose_eval=1)

pred = gbm.predict(X_test)
print ("Accuracy with XGBoost test= ", accuracy_score(pred, y_test))