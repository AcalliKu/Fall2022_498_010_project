import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
import seaborn as sns

def plot_confusion_matrix(actual_val, pred_val, title):
    confusion_matrix = pd.crosstab(actual_val, pred_val,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])
    plot = sns.heatmap(confusion_matrix, annot=True, fmt=',.0f')
    plot.set_title(title)
    plt.show()

# read data
print('reading data...')
NBA_Shot_Logs = pd.read_csv('./data//no_categorical_variables_data.csv')
#NBA_Shot_Logs = pd.read_csv('./data/onehotencoding_shot_logs.csv')
print('finish reading data...')

# assign data and labels
features = NBA_Shot_Logs.columns.values.tolist()
features.pop(0)
features.remove('SHOT_RESULT')
data_X_df = NBA_Shot_Logs[features]
data_Y_df = NBA_Shot_Logs['SHOT_RESULT']
data_X = data_X_df.values.tolist()
data_Y = data_Y_df.values.tolist()

# preprocessing
sc = StandardScaler()
sc.fit(data_X)
data_X = sc.transform(data_X)
# split train/test set
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

# assign parameters for training
C_list = np.logspace(-5, 2, 100)
C_list_exp = np.linspace(0.00001, 100, 100)
scores_record = []
maxC = 0
max_score = 0

# training...Gooooo~~
print('start training')
for i in C_list:
    print(f'case: {i}')
    lr = LogisticRegression(C=i)
    lr.fit(X_train,y_train)
    accuracy = cross_val_score(lr, X_train, y_train, cv=5)
    scores_record.append(accuracy.mean())
    if accuracy.mean() > max_score:
        max_score = accuracy.mean()
        maxC = i

# show training accuracy trace
fig = plt.figure()
plt.xlabel('C')
plt.ylabel('validation accuracy')
plt.plot(C_list_exp, scores_record)
plt.show()

# retrain with best parameter
lr = LogisticRegression(C=maxC)
lr.fit(X_train,y_train)
y_prediction = lr.predict(X_test)

# print result
print(f'bestC: {maxC}')
print(f'coef: {lr.coef_}')
print('start testing')
print('training: ',lr.score(X_train,y_train))
print('testing: ',lr.score(X_test,y_test))

print(confusion_matrix(y_test, y_prediction))
print(f'precision: {precision_score(y_test, y_prediction)}')
print(f'recall: {precision_score(y_test, y_prediction)}')

print ("Accuracy with logistic regression test= ", accuracy_score(y_prediction, y_test))
plot_confusion_matrix(y_test, y_prediction, "Figure 3. LR confusion matrix without ids")