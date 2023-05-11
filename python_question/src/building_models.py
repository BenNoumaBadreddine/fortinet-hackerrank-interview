import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def get_score_of_model(model, x: np.ndarray, y: np.ndarray) -> dict:
    """
    This function evaluate a pretrained model and delivers the accuracy score and the f1 score
    :param model: The pretrained model
    :param x: The input data that we want to evaluate the model with
    :param y: The target values
    :return score_results: The accuracy score and the f1 score
    """
    predictions = model.predict(x)
    model_score = round(accuracy_score(y, predictions), 2) * 100
    f1score = round(f1_score(y, predictions), 2) * 100
    score_results = {"predictions": predictions, "model_score": model_score, "f1score": f1score}
    return score_results


def build_classification_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                               x_valid: np.ndarray, y_valid: np.ndarray,
                               model_name: str = "Logistic Regression", grid_search: bool = True) -> dict:
    """
    This function creates, trains and evaluate the classification model based on the given parameters
    :param x_train: The training input observations
    :param y_train: The training target that we want to predict
    :param x_test: The testing data
    :param y_test: The testing target data
    :param x_valid: The valid data
    :param y_valid: The valid target data
    :param model_name: The name of the model that we want to build
    :param grid_search: A boolean variable that decide either we do grid search for the hyperparameters or no
    :return: results: the result of building, training and evaluating the model.
    """
    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=1)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == "Random Forest":
        if grid_search:
            param_grid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
            model = GridSearchCV(RandomForestClassifier(random_state=1), param_grid)
            model.fit(x_train, y_train)
            # now, we need to get the optimized parameters values
            optimal_max_depth = model.best_estimator_.max_depth
            optimal_n_estimators = model.best_estimator_.n_estimators
            model = RandomForestClassifier(random_state=1, max_depth=optimal_max_depth,
                                           n_estimators=optimal_n_estimators)
        else:
            model = RandomForestClassifier(random_state=1, max_depth=10, n_estimators=50)

    elif model_name == "XGBoost":
        model = XGBClassifier(n_estimators=50, max_depth=4)
    else:
        print(
            f"Warning: The model that you want to build is not yet defined. Please update the "
            f"'build_classification_model' function.")
        print(f"We are going to proceed with the default model for the moment!")
        model = LogisticRegression(random_state=1)

    model.fit(x_train, y_train)
    train_score_results = get_score_of_model(model, x_train, y_train)
    test_score_results = get_score_of_model(model, x_test, y_test)
    valid_score_results = get_score_of_model(model, x_valid, y_valid)

    print(
        f"Results of {model_name} Model on the training dataset: Accuracy = {train_score_results.get('model_score')}"
        f" %, f1_score={train_score_results.get('f1score')}")
    print(
        f"Results of {model_name} Model on the testing dataset: Accuracy = {test_score_results.get('model_score')} "
        f"%, f1_score={test_score_results.get('f1score')}")
    print(
        f"Results of {model_name} Model on the validation dataset: Accuracy = {valid_score_results.get('model_score')} "
        f"%, f1_score={valid_score_results.get('f1score')}")

    results = {"model": model, "train_score_results": train_score_results, "test_score_results": test_score_results,
               "valid_score_results": valid_score_results}
    return results


def create_roc_curve(y_valid: np.ndarray, valid_prediction: np.ndarray):
    """
    This function creates a roc curve of the classification model
    :param y_valid: The target validation data
    :param valid_prediction: The predicted validation data
    :return: plot
    """
    fpr, tpr, _ = metrics.roc_curve(y_valid, valid_prediction)
    auc = round(metrics.roc_auc_score(y_valid, valid_prediction), 2)
    plt.figure(figsize=(3, 3))
    plt.plot(fpr, tpr, label="auc=" + str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.show()
