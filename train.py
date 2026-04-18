from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    data = load_iris()
    X, y = data.data, data.target

    model = RandomForestClassifier()
    model.fit(X, y)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved as model.pkl")

if __name__ == '__main__':
    train_model()