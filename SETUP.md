# Project Setup Guide

## Prerequisites
- Python 3.11.9
- virtualenv package

## Setup Instructions

### 1. Install virtualenv (if not already installed)
```powershell
pip install virtualenv
```

### 2. Create Virtual Environment
Navigate to the project directory and create a virtual environment:

```powershell
cd "c:\Users\Abhay Tyagi\OneDrive - ABES\Desktop\3rd Year Project\3rd-Year-Project"
virtualenv venv
```

**Note:** If Python 3.11.9 is not your default Python, specify the path:
```powershell
virtualenv -p "C:\Path\To\Python311\python.exe" venv
```

Or use the py launcher:
```powershell
py -3.11 -m virtualenv venv
```

### 3. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

**Troubleshooting:** If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### 5. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 6. Verify Installation
```powershell
python -c "import sklearn, xgboost, streamlit, pandas; print('All packages installed successfully!')"
```

## Running the Project

### Train the Model
```powershell
python create_scaler.py
python train_model.py
```

### Run the Streamlit App
```powershell
streamlit run app.py
```

### Deactivate Virtual Environment
When you're done working:
```powershell
deactivate
```

## File Structure
```
3rd-Year-Project/
├── venv/                          # Virtual environment (not tracked by git)
├── archive/                       # Dataset folder (not tracked by git)
│   └── PS_20174392719_1491204439457_log.csv
├── app.py                         # Streamlit web application
├── train_model.py                 # Model training script
├── create_scaler.py              # Scaler creation script
├── splitpy.py                    # Dataset splitting utility
├── model.py                      # Model loading utilities
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git ignore rules
└── Credit_Card_Fraud_Detection_System.ipynb  # Jupyter notebook
```

## Notes
- The `venv/` folder is excluded from git tracking
- All `.pkl` model files are excluded from git tracking
- The `archive/` folder with datasets is excluded from git tracking
- Always activate the virtual environment before working on the project
