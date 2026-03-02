from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and returns a dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error": mean_squared_error(y_test, y_pred)
    }
    
    return metrics