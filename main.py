import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from bayes_opt import BayesianOptimization, UtilityFunction
# from autosklearn.classification import AutoSklearnClassifier


df = pd.read_csv("wine.csv", sep=";")

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

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt,
                   feature_names=X.columns,
                   class_names=['1', '2', '3','4','5','6','7','8'],
                   filled=True)

y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)
print(accuracy_score(y_train, y_train_pred))
print(confusion_matrix(y_train, y_train_pred))

print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

def get_dt_graph(dt_classifier):
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(dt_classifier,
                       feature_names=X.columns,
                       class_names=['1', '2','3','4','5','6','7','8'],
                       filled=True)


def evaluate_model(dt_classifier):
    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))

# Without using any hyperparameterization
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)

gph = get_dt_graph(dt_default)
print('---')
evaluate_model(dt_default)

dt_depth = DecisionTreeClassifier(max_depth=3)
dt_depth.fit(X_train, y_train)
gph = get_dt_graph(dt_depth)
evaluate_model(dt_depth)

dt_min_split = DecisionTreeClassifier(min_samples_split=20)
dt_min_split.fit(X_train, y_train)
gph = get_dt_graph(dt_min_split)
evaluate_model(dt_min_split)

dt_min_leaf = DecisionTreeClassifier(min_samples_leaf=20, random_state=42)
dt_min_leaf.fit(X_train, y_train)
gph = get_dt_graph(dt_min_leaf)
evaluate_model(dt_min_leaf)

dt_min_leaf_entropy = DecisionTreeClassifier(min_samples_leaf=20, random_state=42, criterion="entropy")
dt_min_leaf_entropy.fit(X_train, y_train)
gph = get_dt_graph(dt_min_leaf_entropy)
evaluate_model(dt_min_leaf_entropy)


# text_representation = tree.export_text(model_Decision)
# print(text_representation)



# Using GridSearch For Hyperparameters Optimization

dt = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
grid_search.fit(X_train, y_train)

score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df.head())
print(score_df.nlargest(5,"mean_test_score"))
print(grid_search.best_estimator_)
dt_best = grid_search.best_estimator_
evaluate_model(dt_best)
get_dt_graph(dt_best)
print(classification_report(y_test, dt_best.predict(X_test)))

# For Bayse Optimization

optimizer = BayesianOptimization(f = None,
                                 pbounds = {"C": [0.01, 10],
                                            "degree": [1, 5]},
                                 verbose = 2, random_state = 1234)

utility = UtilityFunction(kind = "ucb", kappa = 1.96, xi = 0.01)
def black_box_function(C, degree):
    # model = SVC(C = C, degree = degree)    # You have to change the model to something that you learned before
    model = dt_default
    model.fit(X_train, y_train)
    # y_score = model.decision_function(X_test)
    evaluate_model(model)
    f = roc_auc_score(y_test, accuracy_score(y_test, dt_default.predict(X_test)))
    return f
# Optimization for loop.
for i in range(25):
    next_point = optimizer.suggest(utility)
    next_point["degree"] = int(next_point["degree"])
    target = black_box_function(**next_point)
    try:
        optimizer.register(params = next_point, target = target)
    except:
        pass
print("Best result: {}; f(x) = {:.3f}.".format(optimizer.max["params"], optimizer.max["target"]))
plt.figure(figsize = (15, 5))
plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
plt.grid(True)
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("Black box function f(x)", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()









# https://www.kaggle.com/code/gauravduttakiit/hyperparameter-tuning-in-decision-trees
# https://medium.com/chinmaygaikwad/hyperparameter-tuning-for-tree-models-f99a66446742
# https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f