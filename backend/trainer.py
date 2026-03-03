import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import os

def train_brain():
    # 1. Create Data (Income, Credit Score, Debt Ratio)
    # 1 = Approved, 0 = Rejected
    data = {
        'income': [50000, 20000, 80000, 15000, 60000, 30000, 100000, 25000],
        'score': [720, 580, 800, 450, 710, 600, 850, 500],
        'debt': [0.1, 0.6, 0.05, 0.8, 0.2, 0.4, 0.02, 0.7],
        'label': [1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    # 2. The Multi-Model Team (Findability Style)
    model1 = RandomForestClassifier(n_estimators=100)
    model2 = XGBClassifier()

    # 3. Combine them for 95%+ accuracy potential
    ensemble = VotingClassifier(
        estimators=[('rf', model1), ('xgb', model2)], 
        voting='soft'
    )

    # 4. Train
    X = df[['income', 'score', 'debt']]
    y = df['label']
    ensemble.fit(X, y)

    # 5. Save the Brain
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(ensemble, 'models/fin_brain.pkl')
    print("✅ Brain Trained and Saved in backend/models/fin_brain.pkl")

if __name__ == "__main__":
    train_brain()