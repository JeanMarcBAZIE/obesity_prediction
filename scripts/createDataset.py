# Telecharger le dataset depuis le site de UCLA
# Dezipper le resulat du telechargement et ne garder que le fichier "*.csv"
# Split en train et test
# Enregistrer dans "assets/data"
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import Config


# Creer les dossier dont on a besoin dans ce script


Config.ORIGINAL_DATA_FILE_PATH.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Telecharge notre fichier dans notre repertoire original pour les données
url="https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic%20(2).zip"
r = requests.get(url, allow_redirects=True)
open(str(Config.ORIGINAL_DATA_FILE_PATH)+'/ObesityDataSet.zip', 'wb').write(r.content)

# Dezipper le resulat du telechargement et ne garder que le fichier "*.csv" et le placer dans les repertoire original et de travail
with ZipFile(str(Config.ORIGINAL_DATA_FILE_PATH)+'/'+str(Config.DATA_DOWNLOAD_ZIP), 'r') as obj_zip:
    FileNames = obj_zip.namelist()
    for fileName in FileNames:
        if fileName.endswith('.csv'):
            obj_zip.extract(fileName,Config.ORIGINAL_DATA_FILE_PATH)
            obj_zip.extract(fileName,Config.DATASET_PATH)

# dataframe
df = pd.read_csv(str(Config.ORIGINAL_DATA_FILE_PATH))

df_train, df_test = train_test_split(
    df, test_size=Config.TEST_SIZE, 
    random_state=Config.RANDON_SEED
)

df_train.to_csv(str(Config.DATASET_PATH / "train.csv"))
df_test.to_csv(str(Config.DATASET_PATH / "test.csv"))