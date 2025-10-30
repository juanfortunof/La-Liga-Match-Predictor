import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.multioutput import MultiOutputRegressor
import seaborn as sns
import matplotlib.pyplot as plt


def scale_data(training_data: pd.DataFrame, test_data: pd.DataFrame, cont_cols: list, cat_cols: list) -> pd.DataFrame:

    ''' This function will return scaled X train, X test, y_train and y test,
        just specify the training data, the testing data and the name of the 
        target column
    '''

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cont_cols),
            ('cat', 'passthrough', cat_cols)],
    remainder='passthrough')

    training_data_preprocessed = preprocessor.fit_transform(training_data)
    test_data_preprocessed = preprocessor.transform(test_data)

    training_data_scaled = pd.DataFrame(training_data_preprocessed, index=training_data.index, columns=cont_cols + cat_cols)
    test_data_scaled = pd.DataFrame(test_data_preprocessed, index=test_data.index, columns=cont_cols + cat_cols)

    for col in training_data_scaled.columns:
        if col in cont_cols: 
            training_data_scaled[col] = training_data_scaled[col].astype(float)
            test_data_scaled[col] = test_data_scaled[col].astype(float)
        else:
            training_data_scaled[col] = training_data_scaled[col].astype(int)
            test_data_scaled[col] = test_data_scaled[col].astype(int)

    return training_data_scaled, test_data_scaled


def train_and_evaluate_xgb(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, multi_output=False, test_set=False):

    """
    Trains an XGBoost classifier using GridSearchCV with a predefined validation set.

    This function is designed for scenarios like time-series or sports data where
    standard K-Fold cross-validation is inappropriate due to potential data leakage.
    It uses a fixed training and validation split for hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        X_val (pd.DataFrame): Validation feature data for tuning and evaluation.
        y_val (pd.Series): Validation target data for tuning and evaluation.

    Returns:
        Tuple[xgb.XGBClassifier, Dict[str, Any], float]: A tuple containing:
            - The best trained XGBoost model object.
            - A dictionary with the best hyperparameters found.
            - The F1 score of the best model on the validation set.
    """

    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)

    # This split index will tell the Grid Search which columns are for training
    # and which ones are for validation, the -1 ones are for training.

    split_index = [-1] * len(X_train) + [0] * len(X_test)
    pds = PredefinedSplit(test_fold=split_index)

    hyperparameters = {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5],
        'min_child_weight': [1, 5],
        'gamma': [0.5, 1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [0.01, 0.1, 1.0],
        'reg_alpha': [0.01, 0.1, 1.0],
        'random_state': [1]
    }

    xgb_classifier = xgb.XGBClassifier()

    if multi_output and test_set:
        xgb_classifier = MultiOutputRegressor(xgb_classifier)
        xgb_classifier.fit(X_train, y_train)
        y_pred = xgb_classifier.predict(X_test)
        return y_pred

    elif multi_output and not test_set:
        xgb_classifier = MultiOutputRegressor(xgb_classifier)

        hyperparameters = {
            'estimator__objective': ['count:poisson'],
            'estimator__eval_metric': ['poisson-nloglik'],
            'estimator__eta': [0.05],
            'estimator__max_depth': [6],
            'estimator__subsample': [0.8],
            'estimator__colsample_bytree': [0.8]}
        

    grid_search_obj = GridSearchCV(
        estimator = xgb_classifier,
        param_grid = hyperparameters,
        cv=pds,
        verbose=1,
        n_jobs=-1
    )

    grid_search_obj.fit(X_combined, y_combined)

    best_model = grid_search_obj.best_estimator_
    best_params = grid_search_obj.best_params_

    print("--- Best Hyperparameters Found ---")
    print(best_params)

    print("\n--- Classification Report on Validation Set ---")
    y_pred = best_model.predict(X_test)

    if multi_output:
        y_pred_proba = None
        feature_imp = None
    else:
        y_pred_proba = best_model.predict_proba(X_test)
        print(classification_report(y_test, y_pred))
        print()

        feature_imp = pd.DataFrame({'Values': best_model.feature_importances_,
                                'Feature': X_train.columns}).sort_values(by='Values', ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(x='Values', y='Feature', data=feature_imp.head(10))
        plt.title('Feature importances')
        plt.show()


    return best_model, best_params, feature_imp, y_pred, y_pred_proba