import xgboost as xgb

def load_model(path='xgb_model.json'):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

model = load_model()

def predict(input_data):
    output = model.predict(input_data)
    return output