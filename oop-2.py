import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    def __init__(self, data_path, target_column, categorical_cols):
        self.data_path = data_path
        self.target_column = target_column
        self.categorical_cols = categorical_cols
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)

    def preprocess_data(self):
        encoded_df = self.df.copy()
        label_encoder = LabelEncoder()
        for col in self.categorical_cols:
            encoded_df[col] = label_encoder.fit_transform(self.df[col])
        encoded_df.drop(columns=['Surname', 'id', 'Unnamed: 0', 'CustomerId'], axis=1, inplace=True)
        encoded_df['CreditScore'] = encoded_df['CreditScore'].fillna(value=encoded_df['CreditScore'].median())
        X = encoded_df.drop(self.target_column, axis=1)
        y = encoded_df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        xgb_model = XGBClassifier()
        xgb_model.fit(self.X_train, self.y_train)
        rf_accuracy = accuracy_score(self.y_test, rf_model.predict(self.X_test))
        xgb_accuracy = accuracy_score(self.y_test, xgb_model.predict(self.X_test))
        if rf_accuracy > xgb_accuracy:
            self.best_model = rf_model
        else:
            self.best_model = xgb_model

    def save_best_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.best_model, f)

# Contoh penggunaan
if __name__ == "__main__":
    trainer = ModelTrainer(data_path="/content/data_A.csv", target_column="churn", categorical_cols=['Geography', 'Gender'])
    trainer.load_data()
    trainer.preprocess_data()
    trainer.train_models()
    trainer.save_best_model("best_model.pkl")