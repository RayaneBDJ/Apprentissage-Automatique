# Projet GIF-4105/7005 Introduction à l'apprentissage automatique (Automne 2023)

Lien vers le [repository Github](https://github.com/louisgreiner/Apprentissage-Automatique).

## Equipe 26

- Rayane Badji (NI: 537 177 500)
- Antoine Gagnon Coulombe (NI: 111 064 130)
- Louis Greiner (NI: 537 160 604)
- Marie Guigon (NI: 537 161 152)
- Charles Hao (NI: 537 160 602)

### Modèles testables

Les modèles suivants ont été implémentés, suivant la base de l'article scientifique, disponible à `pdf/ResLogit-A residual neural network logit model for data-driven choice modelling.pdf` :
```python
models = {
    'mnl': MNLModel(), # MultiNomial Logit
    'mlp-2': MLPModel2(), # MLP network with 2 hidden layers
    'mlp-16': MLPModel16(), # MLP network with 16 hidden layers
    'mlp_2-2': MLP2Model2(), # MLP network with 2 hidden layers, BatchNormalization, Non-Linearity (ReLu) and Dropout
    'mlp_2-16': MLP2Model16(), # MLP network with 16 hidden layers, BatchNormalization, Non-Linearity (ReLu) and Dropout
    'reslogit-2': ResLogitModel(2), # ResLogit model with 2 residual layers
    'reslogit-16': ResLogitModel(16), # ResLogit model with 16 residual layers
    'lmnl-16': LMNLModel() # Learning MultiNomial Logit
}
```

### Jeu de données

Le jeu de données traité est SwissMetro de [Biogeme](https://biogeme.epfl.ch/#data), qui consiste en données de voyageurs et de leur préférences de transport.

L'objectif de ce travail est de pouvoir le transposer à un jeu de données plus grand et plus récent de voyageurs au Canada.

### Biogeme

Biogeme fournit un librairie permettant d'analyser les résultats d'un modèle MultiNomial Logit, que nous n'avons donc que peu utilisé, seulement à fin de tests.

### Comment le tester + enregistrement CSV + display pyplot

Le code permet d'entraîner les différents modèles, sauvegarder les informations relatives au loss, la log-vraisemblance, et l'accuracy des modèles, et enfin les afficher dans des plots.

### Lien vers le poster

Le poster qui sera présenté le mercredi 20 décembre est disponible à `pdf/poster_equipe26.pdf`