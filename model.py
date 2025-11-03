import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Example function to load model and scaler
def load_model_and_scaler():
    # Load the trained XGBoost model
    model = joblib.load('xgboost_model.pkl')

    # Load the StandardScaler
    scaler = joblib.load('scaler.pkl')

    return model, scaler

# Example function to preprocess data
def preprocess_data(data, scaler):
    # Example: scaling numerical features
    numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                          'oldbalanceDest', 'newbalanceDest']
    data[numerical_features] = scaler.transform(data[numerical_features])

    return data

if __name__ == '__main__':
    # Example usage to load model and scaler
    model, scaler = load_model_and_scaler()

    # Example usage to preprocess data
    data = pd.DataFrame({
        'step': [100],
        'type': ['TRANSFER'],
        'amount': [10000],
        'oldbalanceOrg': [50000],
        'newbalanceOrig': [40000],
        'oldbalanceDest': [0],
        'newbalanceDest': [0],
        'isFlaggedFraud': [0]
    })

    # Encode 'type' column using factorize (same as training)
    # TRANSFER=0, PAYMENT=1, CASH_OUT=2, DEBIT=3, CASH_IN=4
    type_mapping = {'TRANSFER': 0, 'PAYMENT': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
    data['type'] = data['type'].map(type_mapping)

    # Preprocess the data (apply scaling)
    data_processed = preprocess_data(data, scaler)

    # Example prediction using the loaded model
    prediction = model.predict(data_processed)

    print("Prediction:", prediction)
    print("Fraud detected!" if prediction[0] == 1 else "Transaction is legitimate.")