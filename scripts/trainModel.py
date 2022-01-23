import pickle # Serialiser des objets (y comporis des modeles) joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

from config import Config


Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)


train_data = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
X_train=train_data[['Gender','Age','Weight','family_history_with_overweight','SCC']]
y_train=train_data[['NObeyesdad']]


#Construction pipeline
numerical_features=['Age','Weight']
categorical_features=['Gender','family_history_with_overweight','SCC']


numerical_pipe=make_pipeline(StandardScaler())
categorical_pipe=make_pipeline(OneHotEncoder())


preprocessor=make_column_transformer((numerical_pipe, numerical_features),(categorical_pipe, categorical_features))

etc_model=make_pipeline(preprocessor, ExtraTreesClassifier(n_estimators=Config.N_ESTIMATOR)) 
etc_model.fit(X_train, y_train.values.ravel())


# Enregisrement du model
pickle.dump(etc_model, open(str(Config.MODELS_PATH / "model.pk"), mode='wb'))