# Credit Card Fraud Detection System

A machine learning-based fraud detection system for online transactions using XGBoost classifier. This project includes comprehensive exploratory data analysis, multiple model comparisons, and a user-friendly Streamlit web application for real-time fraud detection.

## 👥 Team Members
- **Ayush Chauhan**
- **Abhay Tyagi**
- **Adarsh Kumar**
- **Mohammad Noorul Hoda**

## 🎯 Project Overview

This project implements a fraud detection system for financial transactions using machine learning algorithms. The system analyzes transaction patterns and identifies potentially fraudulent activities with high accuracy.

### Key Features
- **Multiple ML Models**: Random Forest, Decision Tree, SVM, and XGBoost classifiers
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Outlier Detection**: Z-score based outlier removal for improved accuracy
- **Web Interface**: Interactive Streamlit application for easy fraud detection
- **Real-time Predictions**: Upload CSV files and get instant fraud detection results

## 📊 Dataset

The project uses the PaySim synthetic financial dataset containing:
- **6,362,620 transactions**
- **11 features** including transaction type, amount, balance information
- **Highly imbalanced dataset** (~0.13% fraud cases)

### Features
- `step`: Time step of transaction
- `type`: Type of transaction (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- `amount`: Transaction amount
- `oldbalanceOrg`: Balance before transaction (origin)
- `newbalanceOrig`: Balance after transaction (origin)
- `oldbalanceDest`: Balance before transaction (destination)
- `newbalanceDest`: Balance after transaction (destination)
- `isFraud`: Target variable (1 = fraud, 0 = legitimate)
- `isFlaggedFraud`: System-flagged suspicious transactions

## 🚀 Getting Started

### Prerequisites
- Python 3.11.9
- virtualenv package
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AbhayTyagi1195/3rd-Year-Project.git
cd 3rd-Year-Project
```

2. **Create virtual environment**
```bash
# Using virtualenv
virtualenv venv

# Or using py launcher for Python 3.11
py -3.11 -m virtualenv venv
```

3. **Activate virtual environment**
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

4. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **Verify installation**
```bash
python -c "import sklearn, xgboost, streamlit, pandas; print('✓ All packages installed successfully!')"
```

## 📁 Project Structure

```
3rd-Year-Project/
├── archive/                                    # Dataset folder (not tracked)
│   └── PS_20174392719_1491204439457_log.csv  # Main dataset
├── venv/                                      # Virtual environment (not tracked)
├── app.py                                     # Streamlit web application
├── train_model.py                             # XGBoost model training script
├── create_scaler.py                           # Feature scaler creation
├── model.py                                   # Model utility functions
├── splitpy.py                                 # Dataset splitting utility
├── Credit_Card_Fraud_Detection_System.ipynb   # Complete analysis notebook
├── requirements.txt                           # Project dependencies
├── SETUP.md                                   # Detailed setup guide
├── .gitignore                                 # Git ignore rules
└── README.md                                  # Project documentation
```

## 🏃 Usage

### Option 1: Using Python Scripts (Recommended for Deployment)

1. **Create the feature scaler**
```bash
python create_scaler.py
```
Output: `scaler.pkl`

2. **Train the XGBoost model**
```bash
python train_model.py
```
Output: `xgboost_model.pkl` and accuracy score

3. **Launch the Streamlit web app**
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### Option 2: Using Jupyter Notebook (Recommended for Analysis)

1. **Start Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open and run**
   - Navigate to `Credit_Card_Fraud_Detection_System.ipynb`
   - Run all cells sequentially
   - Explore EDA, visualizations, and model comparisons

## 🤖 Machine Learning Models

### Models Implemented
1. **Random Forest Classifier**
   - n_estimators: 10
   - max_depth: 10
   - max_features: sqrt

2. **Decision Tree Classifier**
   - max_depth: 10
   - min_samples_split: 10
   - min_samples_leaf: 5

3. **Support Vector Machine (LinearSVC)**
   - C: 1
   - max_iter: 100

4. **XGBoost Classifier** (Best Performance)
   - n_estimators: 10
   - max_depth: 5
   - learning_rate: 0.1

### Model Performance
- XGBoost achieves the highest accuracy
- Handles imbalanced dataset effectively
- Fast inference for real-time predictions

## 📈 Data Preprocessing

### Steps Applied
1. **Data Cleaning**
   - Removal of irrelevant columns (`nameOrig`, `nameDest`, `isFlaggedFraud`)
   - No missing values in dataset

2. **Outlier Removal**
   - Z-score method (threshold: ±3 standard deviations)
   - Applied to numerical features

3. **Feature Engineering**
   - Label encoding for transaction types
   - Feature scaling using StandardScaler

4. **Train-Test Split**
   - 70% training, 30% testing
   - Stratified sampling to maintain class distribution

## 🌐 Web Application Features

### Streamlit App Capabilities
- **File Upload**: Support for CSV files
- **Data Preview**: Display uploaded dataset statistics
- **Fraud Detection**: Real-time predictions on uploaded data
- **Results Dashboard**:
  - Fraud percentage
  - Number of fraudulent transactions
  - Number of legitimate transactions
  - Detailed fraudulent transaction listings

### Required CSV Format
Upload files must contain these columns:
- `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`
- `oldbalanceDest`, `newbalanceDest`, `nameOrig`, `nameDest`
- `isFraud`, `isFlaggedFraud`

## 📊 Visualizations

The Jupyter notebook includes:
- **Univariate Analysis**: Distribution plots, box plots, count plots
- **Bivariate Analysis**: Scatter plots, joint plots
- **Multivariate Analysis**: Correlation heatmaps
- **Model Comparison**: Classification reports, confusion matrices

## 🔧 Advanced Usage

### Split Dataset
Create stratified dataset splits for experimentation:
```bash
python splitpy.py
```
Generates: `dataset_part1.csv`, `dataset_part2.csv`, `dataset_part3.csv`

### Custom Predictions
Use `model.py` for programmatic access:
```python
from model import load_model_and_scaler, preprocess_data

model, scaler = load_model_and_scaler()
# Add your prediction logic here
```

## 📦 Dependencies

Core packages:
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `scikit-learn >= 1.3.0`
- `xgboost >= 2.0.0`
- `streamlit >= 1.28.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `scipy >= 1.11.0`

See `requirements.txt` for complete list with version constraints.

## 🚧 Known Issues

- Streamlit app currently requires `isFraud` column in uploaded CSV (used for evaluation)
- For true inference, remove this requirement in future versions

## 🔮 Future Enhancements

- [ ] Add SMOTE for handling class imbalance
- [ ] Implement real-time API endpoint
- [ ] Add model performance monitoring
- [ ] Include feature importance visualization
- [ ] Support for additional file formats (JSON, Excel)
- [ ] Deploy to cloud platform (AWS/Azure/GCP)
- [ ] Add user authentication and session management

## 📝 License

This project is created for educational purposes as part of a 3rd-year academic project.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or feedback, please reach out to the team members through GitHub.

## 🙏 Acknowledgments

- Dataset: PaySim synthetic financial dataset
- Libraries: scikit-learn, XGBoost, Streamlit
- Academic Institution: ABES (Abhay's affiliation)

---

**Note**: The `archive/` folder containing the dataset and `.pkl` model files are not tracked in Git. You need to run the training scripts after cloning to generate model artifacts.
