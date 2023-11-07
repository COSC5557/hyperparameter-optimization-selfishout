import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization, UtilityFunction
# from autosklearn.classification import AutoSklearnClassifier


df = pd.read_csv("/content/drive/MyDrive/wine.csv", sep=";")

print(df.head(100))

print(df.shape)

print(df.info())

print(df.describe())
print(df.isna().sum())

# Preprocessing

# Splitting the data into training and test sets
X = df.drop('quality',axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=50)
print(X_train.shape)
print(X_test.shape)


# Initiate scaler
sc = StandardScaler()
# Standardize the training dataset
X_train_transformed = pd.DataFrame(sc.fit_transform(X_train),index=X_train.index, columns=X_train.columns)
# Standardized the testing dataset
X_test_transformed = pd.DataFrame(sc.transform(X_test),index=X_test.index, columns=X_test.columns)
# Summary statistics after standardization
X_train_transformed.describe().T

# Check default values
svc = SVC()
params = svc.get_params()
params_df = pd.DataFrame(params, index=[0])
params_df.T



# SVM without Hyperparameter Tuning
model = SVC()
model.fit(X_train, y_train)

# print prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

# Set up score
scoring = ['accuracy']
# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

# fitting the model for grid search
grid.fit(X_train, y_train)


# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)

# print classification report
print(classification_report(y_test, grid_predictions))


# Hyperparameters optimization with Baysian Optimization

# List of C values
C_range = np.logspace(-1, 1, 3)
print(f'The list of values for C are {C_range}')
# List of gamma values
gamma_range = np.logspace(-1, 1, 3)
print(f'The list of values for gamma are {gamma_range}')

# Space
space = {
    'C': hp.choice('C', C_range),
    'gamma': hp.choice('gamma', gamma_range.tolist() + ['scale', 'auto']),
    'kernel': hp.choice('kernel', ['rbf'])}


# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)#

# Objective function
def objective(params):
    svc = SVC(**params)
    scores = cross_val_score(svc, X_train_transformed, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
    # Extract the best score
    best_score = np.mean(scores)
    # Loss must be minimized
    loss = - best_score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# Trials to track progress
bayes_trials = Trials()
# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=bayes_trials)


# Print the index of the best parameters
print(best)
# Print the values of the best parameters
print(space_eval(space, best))


# Train model using the best parameters
svc_bo = SVC(C=space_eval(space, best)['C'], gamma=space_eval(space, best)['gamma'], kernel=space_eval(space, best)['kernel']).fit(X_train_transformed,y_train)
# Print the best accuracy score for the testing dataset
print(f'The accuracy score for the testing dataset is {svc_bo.score(X_test_transformed, y_test):.4f}')










# https://medium.com/grabngoinfo/support-vector-machine-svm-hyperparameter-tuning-in-python-a65586289bcb
# https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# https://medium.com/grabngoinfo/support-vector-machine-svm-hyperparameter-tuning-in-python-a65586289bcb#:~:text=You%20can%20check%20out%20the,to%20make%20it%20linearly%20separable.
