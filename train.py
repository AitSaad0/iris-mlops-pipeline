import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    data = pd.read_csv('data/Iris.csv')

    # Drop useless column
    if 'Id' in data.columns:
        data = data.drop('Id', axis=1)

    # Encode target (text → numbers)
    data['Species'] = data['Species'].astype('category').cat.codes

    # Split
    X = data.drop('Species', axis=1).values
    y = data['Species'].values

    # Train
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved as model.pkl")

if __name__ == '__main__':
    train_model()