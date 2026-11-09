import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    if os.path.exists("tourism_project/data/tourism.csv"):
        csv_path = "tourism_project/data/tourism.csv"
    else:
        csv_path = "content/tourism.csv" 

    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")

    # 1. Handle Duplicate Rows
    df = df.drop_duplicates()

    # 2. Drop Unnecessary Columns
    cols_to_drop = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Data Cleaning & Category Rectifications
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

    # Drop rows with nulls if any
    df = df.dropna()

    # 4. Feature Engineering
    if 'NumberOfPersonVisiting' in df.columns and 'NumberOfChildrenVisiting' in df.columns:
        df['TotalVisitingMembers'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
    if 'MonthlyIncome' in df.columns and 'TotalVisitingMembers' in df.columns:
        df['IncomePerPerson'] = df['MonthlyIncome'] / (df['TotalVisitingMembers'] + 1)

    # 5. Separate Features (X) and Target (y)
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']

    # 6. One-Hot Encoding for categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 7. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensure data output directory exists and save splits
    os.makedirs("tourism_project/data", exist_ok=True)
    X_train.to_csv("tourism_project/data/Xtrain.csv", index=False)
    X_test.to_csv("tourism_project/data/Xtest.csv", index=False)
    y_train.to_csv("tourism_project/data/ytrain.csv", index=False)
    y_test.to_csv("tourism_project/data/ytest.csv", index=False)

    print(" DATA PREPARATION SCRIPT WRITTEN & READY")
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

if __name__ == "__main__":
    prepare_data()
