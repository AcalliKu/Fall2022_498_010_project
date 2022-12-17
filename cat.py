from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import seaborn as sns
import shap

def plot_confusion_matrix(actual_val, pred_val, title):
    confusion_matrix = pd.crosstab(actual_val, pred_val,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])
    plot = sns.heatmap(confusion_matrix, annot=True, fmt=',.0f')
    plot.set_title(title)
    plt.show()

print('reading data...')
NBA_Shot_Logs = pd.read_csv('./data/shot_log_clean.csv')
#NBA_Shot_Logs = pd.read_csv('./no_categorical_variables_data.csv')
print('finish reading data...')

features = NBA_Shot_Logs.columns.values.tolist()
features.pop(0)
features.remove('SHOT_RESULT')

data_X_df = NBA_Shot_Logs[features]
data_Y_df = NBA_Shot_Logs['SHOT_RESULT']
feature_r = features

# calculate percentage of shot made in the dataset
#sns.countplot(x='SHOT_RESULT', data=NBA_Shot_Logs, palette='hls')
#plt.show()

shot_made = 0
for i in data_Y_df:
    if i == 1:
        shot_made +=1
print(f'shot make percentage = {shot_made/len(data_Y_df)}')

# standardize numerical variables
features_non_cat = []
for fea in features:
    if fea != 'player_id' or fea != 'CLOSEST_DEFENDER_PLAYER_ID':
        features_non_cat.append(fea)
data_X_df[features_non_cat] = StandardScaler().fit_transform(data_X_df[features_non_cat])

data_X = data_X_df
data_Y = data_Y_df

# transform categorical variables to string
data_X['CLOSEST_DEFENDER_PLAYER_ID'] = data_X['CLOSEST_DEFENDER_PLAYER_ID'].astype(str)
data_X['player_id'] = data_X['player_id'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)
X_traint, X_val, y_traint, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# hyperparameters tuning
grid = {'iterations': [500, 750, 1000],
        'depth': range(3, 7, 1),
        'learning_rate' : np.linspace(0, 0.2, 5),
        'l2_leaf_reg': range(1, 4, 1)}

estimator = CatBoostClassifier(random_state=42, cat_features = [11, 13])

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=grid,
    scoring = 'accuracy',
    n_jobs = 10,
    cv = 5,
    verbose = 1
)
#grid_search.fit(X_train, y_train)
#print(grid_search.best_params_)


# set model for training/testing
model = CatBoostClassifier(random_state=42,
                         #cat_features = [4, 11, 13],
                         cat_features = [11, 13],
                         use_best_model=True,
                         depth=4,
                         iterations=500,
                         l2_leaf_reg=2,
                         learning_rate=0.1
                        )


classifier = model.fit(X_traint,y_traint, eval_set=(X_val, y_val), verbose=1, plot=False)
y = model.predict(X_train)

# plot shap value summary
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_train)
shap.initjs()
shap.summary_plot(shap_values, X_train, feature_names = features, show=False)
plt.title('Figure 8. CatBoost Features SHAP values')
plt.show()

# testing
preds_class = model.predict(X_test)


feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Figure 7. CatBoost Features Importance')
plt.show()



print(confusion_matrix(y_test,preds_class))
print(precision_score(y_test, preds_class))

print ("Accuracy with CatBoost training= ", accuracy_score(y, y_train))
print ("Accuracy with CatBoost test= ", accuracy_score(preds_class, y_test))
plot_confusion_matrix(y_test, preds_class, "Figure 6. CatBoost confusion matrix")