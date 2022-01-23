""" Considerer train.csv et test.csv pour extraire obtenir des fichiers contenant uniquement les
les variables utiles au modèles et les enregistrer dans un fichier .csv
"""

import pandas as pd 
import numpy as np
from config import Config 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#1 Charger les données depuis le fichier original
Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)
dataset = pd.read_csv(str(Config.ORIGINAL_DATA_FILE_PATH)+'/'+str(Config.DATA_FILE_NAME))

# 2 Supprime les lignes contenant des valeurs manquantes.
dataset.dropna(axis=0, inplace=True)

#3 Encoder les variables qualitative de dataset

le = LabelEncoder()

le.fit(dataset['Gender'].astype(str))
dataset['Gender'] = le.transform(dataset['Gender'].astype(str))

le.fit(dataset['family_history_with_overweight'].astype(str))
dataset['family_history_with_overweight'] = le.transform(dataset['family_history_with_overweight'].astype(str))

le.fit(dataset['FAVC'].astype(str))
dataset['FAVC'] = le.transform(dataset['FAVC'].astype(str))

le.fit(dataset['CAEC'].astype(str))
dataset['CAEC'] = le.transform(dataset['CAEC'].astype(str))

le.fit(dataset['SCC'].astype(str))
dataset['SCC'] = le.transform(dataset['SCC'].astype(str))

le.fit(dataset['CALC'].astype(str))
dataset['CALC'] = le.transform(dataset['CALC'].astype(str))

le.fit(dataset['MTRANS'].astype(str))
dataset['MTRANS'] = le.transform(dataset['MTRANS'].astype(str))

le.fit(dataset['SMOKE'].astype(str))
dataset['SMOKE'] = le.transform(dataset['SMOKE'].astype(str))

le.fit(dataset['NObeyesdad'].astype(str))
dataset['NObeyesdad'] = le.transform(dataset['NObeyesdad'].astype(str))


#4 Selection grace au Model pour retourner le nombre idéal de variables 
features= list(dataset.columns[0:16])

X=dataset[features]
y=dataset["NObeyesdad"]
clf = ExtraTreesClassifier()
#clf=DecisionTreeClassifier()
clf = clf.fit(X, y)

# Selection des 5 variables les plus importantes avec le highest chi-squared le plus elevés
selector=SelectKBest(chi2, k=5)
selector.fit_transform(X,y)
selector.get_support()


#Splitter les données avec uniquement les variables utiles et la target
features_names=dataset.columns[0:16]
dataset.columns[0:16]
final_features=np.array(features_names)[selector.get_support()]
final_features = final_features.tolist()
type(final_features)
final_features.append("NObeyesdad")


#Fichier final des features plus la target
train_features=pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
train_features=train_features[final_features]

test_features=pd.read_csv(str(Config.DATASET_PATH/"test.csv"))
test_features=train_features[final_features]

train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)