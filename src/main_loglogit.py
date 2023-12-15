import logging
import math
from biogeme import models, biogeme as bio, expressions as be
from datasets.swissmetro_biogeme import (
    database,
    ASC_TRAIN, 
    B_TIME, 
    TRAIN_TT_SCALED, 
    B_COST,
    TRAIN_COST_SCALED,
    ASC_SM,
    SM_TT_SCALED,
    SM_COST_SCALED,
    ASC_CAR,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    CAR_AV_SP)

# Definition of the utility functions
V1 = ASC_TRAIN + \
     B_TIME * TRAIN_TT_SCALED + \
     B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + \
     B_TIME * SM_TT_SCALED + \
     B_COST * SM_COST_SCALED
V3 = ASC_CAR + \
     B_TIME * CAR_TT_SCALED + \
     B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP,
      2: be.Variable('SM_AV'),
      3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, be.Variable('CHOICE'))

# Define level of verbosity
logger = logging.getLogger('biogeme')
# Détermine le niveau de détail retourné durant l'exécution.
logger.setLevel(logging.INFO)

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = "b01logit" # Nom du rapport. Arbitraire.
biogeme.calculateNullLoglikelihood(av) # Métrique de plus. Recommandé par le manuel.

# Estimate the parameters
results = biogeme.estimate()
print(results.short_summary())


# Print the estimated values
betas = results.getBetaValues()
for k, v in betas.items():
    print(f"{k:10}=\t{v:.3g}")

# Get the results in a pandas table
# Les résulats obtenus sont négatifs et je sais pas pourquoi. Il faudra investiguer comment interpréter les résultats.
pandasResults = results.getEstimatedParameters() 
print(pandasResults)
values = pandasResults['Value']
for val in values[0:2]:
    print(1 / (1 + math.e ** val ))
