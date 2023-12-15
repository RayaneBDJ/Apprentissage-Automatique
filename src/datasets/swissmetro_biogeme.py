import pandas as pd
from biogeme import database as db, expressions as be


df = pd.read_csv("data/swissmetro.dat", sep='\t')
database = db.Database("swissmetro", df)
# On peut obtenir un array numpy à partir de pandas avec df.values

# Removing some observations can be done directly using pandas.
# on enlève PURPOSE = Commute et Business (on se focalise les choix hors travail) et Choice = unknown (car apporte aucune information)
remove = (((database.data.PURPOSE != 1) & (database.data.PURPOSE != 3)) | (database.data.CHOICE == 0))
database.data.drop(database.data[remove].index, inplace=True)

# Parameters to be estimated
ASC_CAR = be.Beta('ASC_CAR', 0, None, None, 1)
ASC_TRAIN = be.Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = be.Beta('ASC_SM', 0, None, None, 0) # Le 1 veut dire que la variable est figée et ne sera pas modifiée par le modèle
B_TIME = be.Beta('B_TIME', 0, None, None, 0)
B_COST = be.Beta('B_COST', 0, None, None, 0)

# Definition of new variables
# on enlève les personnes qui ont un abonnement de train/métro annuel, car forcément leur choix final est biaisé (= choisir ce pour quoi ils paient à l'année)
SM_COST = be.Variable('SM_CO') * (be.Variable('GA') == 0) # GA = 1 => la personne a un abonnement annuel, 0 => non
TRAIN_COST = be.Variable('TRAIN_CO') * (be.Variable('GA') == 0)

# Definition of new variables: adding columns to the database
CAR_AV_SP = database.DefineVariable('CAR_AV_SP', be.Variable('CAR_AV') * (be.Variable('SP') != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', be.Variable('TRAIN_AV') * (be.Variable('SP') != 0))
TRAIN_TT_SCALED = database.DefineVariable('TRAIN_TT_SCALED', be.Variable('TRAIN_TT') / 100.0)
TRAIN_COST_SCALED = database.DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100)
SM_TT_SCALED = database.DefineVariable('SM_TT_SCALED', be.Variable('SM_TT') / 100.0)
SM_COST_SCALED = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.DefineVariable('CAR_TT_SCALED', be.Variable('CAR_TT') / 100)
CAR_CO_SCALED = database.DefineVariable('CAR_CO_SCALED', be.Variable('CAR_CO') / 100)