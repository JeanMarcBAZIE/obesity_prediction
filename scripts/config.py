from pathlib import Path

class Config:
	TEST_SIZE=0.2
	ASSETS_PATH=Path("../assets")
	ORIGINAL_DATA_FILE_PATH=ASSETS_PATH/"data_base"    # Les données d'origines
	DATASET_PATH=ASSETS_PATH/"data"		               # Dossier pour la dataset avec laquelle nous allons travailler
	DATA_DOWNLOAD_ZIP="ObesityDataSet.zip"        # Chemin du fichier zip telecharger sur le site de UCLA
	DATA_FILE_NAME="ObesityDataSet.csv"
	MODELS_PATH=ASSETS_PATH/"models"               		   # Dossier pour les models
	FEATURES_PATH=ASSETS_PATH/"features"      		   # Dossier pour les features
	METRICS_FILE_PATH=ASSETS_PATH/"metrics"                     #Fichiers pour les metrics
	N_ESTIMATOR=100
    



