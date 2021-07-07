import joblib
import numpy as np
import os


def init():
    global model
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], 'model.pkl')
    model = joblib.load(model_path)


def run(request):
    text = json.loads(request)
    data = pd.DataFrame(text['data'])
    
    result = model.predict(data)
    return result.tolist()
    