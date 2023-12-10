import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, StratifiedShuffleSplit
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
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
# from bayes_opt import BayesianOptimization, UtilityFunction
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



sc = StandardScaler()
# Standardize the training dataset
X_train_transformed = pd.DataFrame(sc.fit_transform(X_train),index=X_train.index, columns=X_train.columns)
# Standardized the testing dataset
X_test_transformed = pd.DataFrame(sc.transform(X_test),index=X_test.index, columns=X_test.columns)
# Summary statistics after standardization
X_train_transformed.describe().T


# Random Forest without Hyperparameter Tuning
model = RandomForestClassifier(random_state=50)
model.fit(X_train, y_train)
baseline_predictions_train = model.predict(X_train)
baseline_accuracy_train = accuracy_score(y_train, baseline_predictions_train)
print(f"Baseline Training Accuracy: {baseline_accuracy_train:.4f}")


predictions = model.predict(X_test)
print(f"Predictions of Random Forest without Hyperparameter TRuning: {classification_report(y_test, predictions)}")
accuracy_baseline = accuracy_score(y_test, predictions)
print(f"Accuracy_baseline: {accuracy_baseline:.4f}")
score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"Accuracy_baseline: {score:.4f}")
f1_without = f1_score(y_test, predictions, average='weighted')
print(f"F1 Score: {f1_without:.4f}")


param_rand = {
    'n_estimators':  [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_split':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'max_features':['sqrt', 'log2', None],
}

n_iter_search = 100
# random_search = RandomizedSearchCV(RandomForestClassifier(random_state=50), param_distributions=param_rand, n_iter=n_iter_search,
#     cv=5,               
#     random_state=42,
#     n_jobs=-1           
# )
# random_search.fit(X_train, y_train)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# List to store each outer fold's accuracy
outer_fold_accuracies = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train1, X_test1 = X.iloc[train_idx], X.iloc[test_idx]
    y_train1, y_test1 = y.iloc[train_idx], y.iloc[test_idx]

    # Inner CV and Random Search
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(model, param_distributions=param_rand, n_iter=10, cv=inner_cv)
    random_search.fit(X_train1, y_train1)


    outer_fold_accuracy = random_search.score(X_test1, y_test1)
    # outer_fold_accuracy = random_search.predict(X_test1)
    outer_fold_accuracies.append(outer_fold_accuracy)

print("Outer_fold_accuracues: ")
print(outer_fold_accuracies)

plt.boxplot(outer_fold_accuracies, patch_artist=True)
plt.title('Distribution of Accuracies Across Outer Folds')
plt.ylabel('Accuracy')
plt.xticks([1], ['Random Forest'])
plt.show()




# Print best parameters after tuning
print(random_search.best_params_)

# Print how our model looks after hyper-parameter tuning
print(random_search.best_estimator_)

random_predictions = random_search.predict(X_test)

# Print classification report
print(f"Prediction of Random Forest with GridSeaech: {classification_report(y_test, random_predictions)}")
accuracy_random = accuracy_score(y_test, random_predictions)
print(f"Accuracy_Grid: {accuracy_random:.4f}")


random_predictions_train = random_search.best_estimator_.predict(X_train)

# Calculate training accuracy
random_accuracy_train = accuracy_score(y_train, random_predictions_train)

print(f"Grid Search Training Error: {random_accuracy_train:.4f}")



# Space for Bayesian Optimization Random Forest
space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
    'max_depth': hp.choice('max_depth',[5, 10, 15, 20, 25, 30]),
    'min_samples_split': hp.choice('min_samples_split', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),

}
scoring = ['accuracy']

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# Objective function
# def objective(params):

#     rf_model = RandomForestClassifier(**params)
#     scores = cross_val_score(rf_model, X_train_transformed, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
#     best_score = np.mean(scores)
#     loss = - best_score
#     return {'loss': loss, 'params': params, 'status': STATUS_OK}
# bayes_trials = Trials()
# # Optimize
# best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = bayes_trials)

# # Print the index of the best parameters
# print(best)
# # Print the values of the best parameters
# print(space_eval(space, best))

def objective(params):
    clf = RandomForestClassifier(**params)
    # Inner cross-validation
    score = cross_val_score(clf, X_inner, y_inner, scoring='accuracy', cv=3).mean()
    return {'loss': -score, 'status': STATUS_OK}


for train_idx, test_idx in outer_cv.split(X, y):
    X_inner, X_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_inner, y_outer = y.iloc[train_idx], y.iloc[test_idx]

    # Bayesian Optimization in the inner loop
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

    # Train model on the best parameters
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_inner, y_inner)

    Evaluate on the outer loop test set
    outer_fold_accuracy = best_model.score(X_outer, y_outer)
    outer_fold_accuracies.append(outer_fold_accuracy)

print("Outer_fold_accuracues: ")
print(outer_fold_accuracies)

plt.boxplot(outer_fold_accuracies, patch_artist=True)
plt.title('Distribution of Accuracies Across Outer Folds')
plt.ylabel('Accuracy')
plt.xticks([1], ['Random Forest'])
plt.show()






# Train model using the best parameters
best_params = space_eval(space, best)
rf_bo = RandomForestClassifier(**best_params, random_state=50).fit(X_train, y_train)
print(f'The accuracy score for the testing dataset is {rf_bo.score(X_test, y_test):.4f}')



bo_predictions_train = rf_bo.predict(X_train)

# Calculate training accuracy
bo_accuracy_train = accuracy_score(y_train, bo_predictions_train)
print(f"Bayesian Optimization Training Accuracy: {bo_accuracy_train:.4f}")


train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X_train, y_train, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.show()




def plot_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Assuming 'best_params' contains the best parameters found by Bayesian Optimization
best_model = RandomForestClassifier(**best_params, random_state=50)

# Plotting the learning curve
plot_learning_curve(best_model, X, y, ylim=(0.1, 1.01), cv=5, n_jobs=4)


def calculate_accuracies(model, X, y, train_sizes):
    accuracies = []

    for size in train_sizes:
        test_size = 1 - size
        # Ensure test_size is not zero
        if test_size <= 0.1:  # Set a minimum test size
            test_size = 0.1

        sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=42)
        score = cross_val_score(model, X, y, cv=sss, scoring='accuracy').mean()
        accuracies.append(score)

    return accuracies

train_sizes = np.linspace(0.1, 1.0, 10)
baseline_accuracies = calculate_accuracies(model, X, y, train_sizes)
bayesian_accuracies = calculate_accuracies(best_model, X, y, train_sizes)


plt.figure(figsize=(10, 6))
plt.plot(train_sizes, baseline_accuracies, label='Baseline Model', marker='o')
plt.plot(train_sizes, bayesian_accuracies, label='Bayesian Optimization Model', marker='o')
plt.title('Model Accuracy Comparison')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()








# https://medium.com/grabngoinfo/support-vector-machine-svm-hyperparameter-tuning-in-python-a65586289bcb
# https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# https://medium.com/grabngoinfo/support-vector-machine-svm-hyperparameter-tuning-in-python-a65586289bcb#:~:text=You%20can%20check%20out%20the,to%20make%20it%20linearly%20separable.
