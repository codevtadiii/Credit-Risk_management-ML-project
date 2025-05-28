import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath):
    
    print("Loading dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    print("Dataset loaded.")

    
    selected_cols = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'emp_length',
        'home_ownership', 'annual_inc', 'verification_status', 'purpose', 
        'addr_state', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
        'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'loan_status'
    ]
    df = df[selected_cols]
    
    df = df.dropna()


    
    default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']
    df['default'] = df['loan_status'].apply(lambda x: 1 if x in default_statuses else 0)

    
    df['default_amount'] = df['loan_amnt'] * df['default']

    
    df.drop(columns=['loan_status'], inplace=True)

    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(['default', 'default_amount'])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    
    X = df.drop(columns=['default', 'default_amount'])
    y_class = df['default']
    y_reg = df['default_amount']

    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    return X_train_c, X_test_c, y_train_c, y_test_c, X_train_r, X_test_r, y_train_r, y_test_r

