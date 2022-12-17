import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import time
import shap
import seaborn as sns

def plot_confusion_matrix(actual_val, pred_val, title):
    confusion_matrix = pd.crosstab(actual_val, pred_val,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])
    plot = sns.heatmap(confusion_matrix, annot=True, fmt=',.0f')
    plot.set_title(title)
    plt.show()

print('reading data...')
#NBA_Shot_Logs = pd.read_csv('./shot_log_clean.csv')
#NBA_Shot_Logs = pd.read_csv('./data/no_categorical_variables_data.csv')
NBA_Shot_Logs = pd.read_csv('./data/onehotencoding_shot_logs.csv')
print('finish reading data...')

features = NBA_Shot_Logs.columns.values.tolist()
features.pop(0)
features.remove('SHOT_RESULT')

print(NBA_Shot_Logs.shape)

# one hot encoding (moved to preprocessing)
#transformer = make_column_transformer(
#    (OneHotEncoder(), ['PERIOD']), remainder='passthrough')
#    #'CLOSEST_DEFENDER_PLAYER_ID', 'player_id',
#transformed = transformer.fit_transform(NBA_Shot_Logs)
#transformed_dense = csr_matrix.toarray(transformed)
#NBA_Shot_Logs = pd.DataFrame(transformed, columns=transformer.get_feature_names())
#features = ['SHOT_DIST','TOUCH_TIME','FINAL_MARGIN','PERIOD','SHOT_CLOCK','DRIBBLES','CLOSE_DEF_DIST']



data_X_df = NBA_Shot_Logs[features]
data_Y_df = NBA_Shot_Logs['SHOT_RESULT']

data_X = data_X_df.values.tolist()
data_Y = data_Y_df.values.tolist()

sc = StandardScaler()
sc.fit(data_X)
data_X = sc.transform(data_X)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

dtest = xgb.DMatrix(X_test)
dtrain_check = xgb.DMatrix(X_train)
d_train_xgboost = xgb.DMatrix(X_train,label = y_train)


estimator = xgb.XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)

# hyperparameters tuning

parameters = {
    'eta': np.linspace(0.01, 0.2, 5),
    'min_child_weight': np.logspace(-7, -3, 4),
    'max_depth': range(3, 7, 1),
    #'gamma': np.linspace(0, 1, 5),
    'n_estimators': range(60, 220, 40)
    #'subsample': np.linspace(0.5, 1, 5),
    #'colsample_bytree': np.linspace(0.5, 1, 5)
}



grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'accuracy',
    n_jobs = 10,
    cv = 5,
    verbose = 3
)
start = time.time()

#grid_search.fit(X_train, y_train)
#print('grid_search time = %.1f sec'%(time.time() - start))
#print(grid_search.best_estimator_)


# train with best parameters
parameters_xgb ={
            'objective': 'binary:logistic',
            'max_depth':3,
            'min_child_weight': 1e-07,
            'eval_metric':'auc',
            'eta':0.104999997,
            'random_state': 42,
            }

xgb_model = xgb.train(parameters_xgb, d_train_xgboost, 100)
y_train_check = xgb_model.predict(dtrain_check)
y_pred_xgb = xgb_model.predict(dtest)

## plot shap values
#explainer = shap.TreeExplainer(xgb_model)
#shap_values = explainer.shap_values(X_train)
#print(type(shap_values))
#shap.initjs()
#shap.summary_plot(shap_values, X_train, feature_names = features)
#plt.show()

# turn output to binary result
for i in range(0, y_train_check.shape[0]):
    if y_train_check[i] >= 0.5:
       y_train_check[i]=1
    else:
       y_train_check[i]=0

for i in range(0, y_pred_xgb.shape[0]):
    if y_pred_xgb[i] >= 0.5:
       y_pred_xgb[i]=1
    else:
       y_pred_xgb[i]=0

print(confusion_matrix(y_test,y_pred_xgb))
print(f'precision: {precision_score(y_test, y_pred_xgb)}')
print(f'recall: {precision_score(y_test, y_pred_xgb)}')

print ("Accuracy with XGBoost train= ", accuracy_score(y_train_check, y_train))
print ("Accuracy with XGBoost test= ", accuracy_score(y_pred_xgb, y_test))

plot_confusion_matrix(y_test, y_pred_xgb, "Figure 5. XGBoost confusion matrix")

ax = xgb.plot_importance(xgb_model, max_num_features = 8)
fig = ax.figure
# plt.show()
fig.savefig("importance.png")