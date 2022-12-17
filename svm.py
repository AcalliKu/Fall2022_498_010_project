from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import time
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(actual_val, pred_val, title):
    confusion_matrix = pd.crosstab(actual_val, pred_val,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])
    plot = sns.heatmap(confusion_matrix, annot=True, fmt=',.0f')
    plot.set_title(title)
    plt.show()

print('reading data...')
#NBA_Shot_Logs = pd.read_csv('./data/shot_log_clean.csv')
NBA_Shot_Logs = pd.read_csv('./data/no_categorical_variables_data.csv')
print('finish reading data...')

features = NBA_Shot_Logs.columns.values.tolist()
features.pop(0)
features.remove('SHOT_RESULT')

data_X_df = NBA_Shot_Logs[features]
data_Y_df = NBA_Shot_Logs['SHOT_RESULT']

data_X = data_X_df.values.tolist()
data_Y = data_Y_df.values.tolist()

sc = StandardScaler()
sc.fit(data_X)
data_X = sc.transform(data_X)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.4, random_state=42)
X_train_t, X_val, y_train_t, y_val = train_test_split(X_train, y_train , test_size=0.4, random_state=42)


C_grid = np.logspace(-3, 3, 10)
gamma_grid = np.logspace(-3, 3, 10)
scores_record = []
maxC = 0
max_gamma = 0
max_score = 0

start = time.time()
#print('start training')
#for i in C_grid:
#    for j in gamma_grid:
#        print(f'C = {i}, gamma = {j}')
#        clf= svm.SVC(C=i, kernel='rbf', gamma=j)
#        clf.fit(X_train_t,y_train_t)
#
#        #cross validation
#        #accuracy = cross_val_score(clf, X_train, y_train, cv=5)
#        #scores_record.append(accuracy.mean())
#        #if accuracy.mean() > max_score:
#        #    max_score = accuracy.mean()
#        #    maxC = i
#        #    max_gamma = j
#        accuracy_val = clf.score(X_val, y_val)
#        scores_record.append(accuracy_val)
#        if accuracy_val > max_score:
#            max_score = accuracy_val
#            maxC = i
#            max_gamma = j
#        print(f'best C: {maxC}, best gamma: {max_gamma}, accuracy: {max_score}')
#
#print(f'best C: {maxC}, best gamma: {max_gamma}, accuracy: {max_score}')



print('training time = %.1f sec'%(time.time() - start))
start = time.time()
# retrain
maxC = 1000
max_gamma = 0.004641588833612777
clf= svm.SVC(C=maxC, kernel='rbf', gamma=max_gamma)
clf.fit(X_train,y_train)
#y_train_prediction = clf.predict(X_train)
y_prediction = clf.predict(X_test)


print(confusion_matrix(y_test, y_prediction))
print(f'precision: {precision_score(y_test, y_prediction)}')
print(f'recall: {precision_score(y_test, y_prediction)}')
#print ("Accuracy with SVM train= ", accuracy_score(y_train_prediction, y_train))
print ("Accuracy with SVM test= ", accuracy_score(y_prediction, y_test))
plot_confusion_matrix(y_test, y_prediction, 'Figure 4. SVM confusion matrix')

print('testing time = %.1f sec'%(time.time() - start))