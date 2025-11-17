from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [2000]
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
