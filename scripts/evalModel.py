import pickle # Serialiser des objets (y comporis des modeles)
import json 

import pandas as pd
from sklearn.metrics import accuracy_score
from config import Config

Config.METRICS_FILE_PATH.mkdir(parents=True, exist_ok=True)

test_data = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
X_test=test_data[['Gender','Age','Weight','family_history_with_overweight','SCC']]
y_test=test_data[['NObeyesdad']]

y_test = y_test.to_numpy().ravel()


model = pickle.load(open(str(Config.MODELS_PATH / "model.pk"), mode='rb'))

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

with open(str(Config.METRICS_FILE_PATH)+"/result.json", mode='w') as f:
    json.dump(dict(test_accuracy=test_accuracy), f)