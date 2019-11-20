# Réalisation de la tâche du Wisconsin par force learning

## Test rapide :

Avec 3 dimentions et 4 cartes à jouer : 
```
python3 runme.py
```
Avec vos propres paramètres (doivent être des entiers)
```
python3 runme.py dimension nbrCards
```

## Contenu du projet

### Description des fichiers

* README.md : ce fichier
* experiment.py : le fichier de la fonction à faire tourner dans hyperperopt, générant un reservoir et le testant
* gener.py : fonction générant une entrée pour le jeu de données
* hp_analyse_error : analyser les données obtenues par hyperopt, va retourner des graphiques
* hp_experiment.py : permet de déterminer les intervalles d'évaluation sous hyperopt, et de lancer le calcul
* model_memory : modèle du reservoir sous force learning avec le reward pouvant varier
* model_memory_V2 : modèle du reservoir sous force learning avec le reward fixé
* runme.py : lancer une execution du projet avec des hyperparamètres précis
* createGraph.py : créer les graphiques d'analyse d'activités du reservoir 

### Description des dossiers

* example : fichiers créés avec createGraph.py
* final : fichiers créés avec hp_experiment.py
* params : contient un fichier de paramètre à utiliser par défaut dans hyperopt
* params_hp : contient les paramètres par défaut utilisés par une exécution d'hyperopt

## Exemple de calcul, d'analyse et de génération de graphiques

1) Choisir quel type de modèle utiliser : changer la première ligne de experiment.py ou createGraph.py en
from model_memory_fixed import Force OU from model_memory import Force
Puis choisir dans ces mêmes fichiers la partie créant le modèle à dé-commenter
2) Choisir les hyperparamètres à explorer dans le fichier hp_experiment.py dans la partie EXPLORE PARAMETERS
Ces mêmes paramètres doivent être choisir en argument dans le fichier experiment.py, les autres doivent être fixer
3) Pour la recherche d'hyperparmètres, lancer :
```
python3 hp_experiment.py -f EN_HP_default_params_sent_comprehension.json -s nom_execution
```
4) Lancer la création des graphiques d'analyse :
```
 python3 hp_analyse_error.py -l -t ./final/choix_execution/hyperopt_trials.pkl -m
```
5) Après analyse, mettre à jour les hyperparamètres voulus dans le fichier runme.py
6) Les graphiques d'activités se trouvent dans le dossier example

## Compréhension des hyperparamètres

### Paramètres généraux

Les paramètres généraux sont :
* nb_dimension : Nombre de dimension du jeu
* nbrcartes : Nombre de carte possible sur le plateau en dehors de la carte de test

### Jeu de données

On peut choisir comment va varier le jeu de données avec les paramètres suivants :
* datasize : la taille du jeu de donnée
* volatility : change la règle cachée à chaque fois que taille(jeu_donnée)%volatility == 0

Afin de faire varier la volatility on modifie la **variability**, et ainsi que le réseau n'apprenne pas en sachant que volatility est fixe
On obtient donc : 
addVariability = randint(-variability,variability)
taille(jeu_donnée)%(volatility+addVariability) == 0

### Temps de calcul

Certains hyperparamètres influent sur le temps de calcul, plus ils sont faibles et plus de temps de calcul le sera aussi,
influant aussi les performances du reservoir.
La liste des ces hyperparamètres :
* datasize : la taille du jeu de donnée
* sizeres : la taille du reservoir
* stimulus_length_training : Durée d'apprentissage d'un input
* _stimulus_length_warm_up : Durée de mise à jour du reservoir, où il tourne dans le vide
* stimulus_updating_reward : Phase durant laquelle le reward est update
* inst_number : Nombre d'instance réalisée (créer une moyenne plus pertinente)

### Force learning et autres

**alpha** est le coeffiscient de regularisation pour l'apprentissage,
il est utilisé en début d'apprentissage lors de la création de la matrice identité afin d'avoir la matrice P

**leak** est le leaking rate du reservoir, plus cette valeur est proche de 1 et mois les executions précédentes seront 
prise en compte à l'instant t

**noise** est le bruit à ajouter aux sorties du reservoir, plus cette valeur est faible et moins il y a de bruit

### Scaling et sparsity
Les hyperparamètres à **scaling** permettent de déterminer l'impact sur le reservoir de certains poids
Le **spectral_radius** est le scaling du reservoir
Les hyperparamètres à **sparsity** permettent de déterminer le nombre de connexion au reservoir 

## Requirement

```
Python 3.5.2
tqdm==4.31.1
matplotlib==3.0.2
numpy==1.16.1
hyperopt==0.1.2
```

## Auteurs :
Principal:
* **Raphael TEITGEN** ( raphael.teitgen@inria.fr )
Maitres de stage
* **Frederic ALEXANDRE** ( frederic.alexandre@inria.fr )
* **Xavier HINAUT** ( xavier.hinaut@inria.fr )
Soutiens :
* **Anthony STROCK** (anthony.strock@inria.fr )
