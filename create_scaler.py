import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset
data = pd.read_csv('archive/PS_20174392719_1491204439457_log.csv')

# Drop non-relevant columns
data.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Encode 'type' column if it's categorical
data['type'] = pd.factorize(data['type'])[0]

# Select numerical features for scaling
numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                      'oldbalanceDest', 'newbalanceDest']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the numerical features
scaler.fit(data[numerical_features])

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')