import pandas as pd
import numpy as np

# Pour Sklearn (Non testé)
def load_numpy_data():
    # Notre fichier de données
    df = pd.read_csv("data/swissmetro.dat", sep='\t')
    # Nous retirons les cas PURPOSE = 1 et PURPOSE = 3 qui signifient des choix autres que ceux qui nous intéressent. Choice = 0 est une donnée manquante, donc on retire
    df.drop(df[((df['PURPOSE'] != 1) & (df['PURPOSE'] != 3)) | (df['CHOICE'] == 0)].index, inplace=True) 

    # On enlève la colonne CHOICE, qui correspond à notre vecteur r
    X = df.drop('CHOICE', axis=1).values.astype(np.float32)
    # Décalage de 1 pour avoir les classes 0, 1 et 2
    y = df['CHOICE'].values - 1

    return X, y