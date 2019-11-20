
# 1 seul reward par stimulus (la reward n'est calculé qu'une seule fois par chaque stimulus)
MODEL_FIXED = True
from model_memory_fixed import Force

# la reward est calculé/updaté à chaque fois qu'un choix est fait (à tous les pas de temps)
# équivalent à avoir le droit à plusieurs essais (choix de carte) pour chaque stimulus
# from model_memory import Force
# MODEL_FIXED = False

import gener as gen
from random import randint,shuffle
import numpy as np
from functools import partial


def Experiment(args):

    volatility = 5
    variability = 3
    data_size = 300
    nb_dimension=3
    nbrcartes= 4
    noise = 1e-3
    sizeres = 100
    inst_number = 1

    """
    leak = args['leak']
    alpha = args['alpha']
    fbscale = args['fbscale']
    inputscale = args['inputscale']
    rewardscale = args['rewardscale']
    spectral_Radius = args['spectral_Radius']
    sparsity = args['sparsity']
    sparsityin = args['sparsityin']
    sparsityrw = args['sparsityrw']
    sparsitychoice = args['sparsitychoice']
    choicescale = args['choicescale']
    """
    stimulus_length_training = int(args["stimulus_length_training"])
    _stimulus_length_warm_up = int(args["_stimulus_length_warm_up"])
    stimulus_updating_reward = int(args["stimulus_updating_reward"])

    leak = 0.6
    alpha = 1e-6
    fbscale = 1e-3
    inputscale = 1.
    rewardscale = 5.
    spectral_Radius = 1e-4
    sparsity = 0.65
    sparsityin = 0.60
    sparsityrw = 0.8
    sparsitychoice = 0.9
    choicescale = 5.0
    #stimulus_length_training = 4
    #_stimulus_length_warm_up=2

    #Data creation
    features = []
    labels = []

    #definitive choice of the hidden rule
    value = 0

    #create the first number of data generated with the first rule
    rd_variability = randint(-variability,variability)
    #Creation of features and labels
    while len(features)<data_size:

        if len(features) % (volatility+rd_variability) == 0 and len(features)>0:
            value=nb_dimension-1
            rd_variability = randint(-variability,variability)

        feat,label = gen.carte(nb_dimension,nbrcartes,value)
        labels.append(label)
        features.append(feat)

    size_input = nbrcartes * nb_dimension
    label_size = nbrcartes

    data = np.zeros(data_size, dtype = [ ("input",  float, (size_input,)),
                                         ("output", float, (label_size,))])
    data["input"] = features
    data["output"] = labels

    #split the dataset in 2 part
    train_data = data[:int(2/3*len(data))]
    test_data = data[int(len(data)*2/3):]

    result = []
    for rs in range(inst_number):

        #----CREATION OF THE MODEL----

        if MODEL_FIXED:
            #For model_memory_fixed
            model = Force(shape=(size_input,sizeres,label_size),
            spectral_Radius=spectral_Radius,sparsity=sparsity,leak=leak,noise=noise,
            inputscale = inputscale, fbscale = fbscale,
            stimulus_length_training=stimulus_length_training,alpha=alpha,rewardscale=rewardscale,_stimulus_length_warm_up=_stimulus_length_warm_up,
            sparsityin=sparsityin,sparsityrw=sparsityrw,
            sparsitychoice=sparsitychoice,choicescale=choicescale,stimulus_updating_reward=stimulus_updating_reward)
        else:
            #For model_memory
            model = Force(shape=(size_input,sizeres,label_size),
            spectral_Radius=spectral_Radius,sparsity=sparsity,leak=leak,noise=noise,
            inputscale = inputscale, fbscale = fbscale,
            stimulus_length_training=stimulus_length_training,alpha=alpha,rewardscale=rewardscale,_stimulus_length_warm_up=_stimulus_length_warm_up,
            sparsityin=sparsityin,sparsityrw=sparsityrw,
            sparsitychoice=sparsitychoice,choicescale=choicescale)
        model.train(train_data)
        error = model.test(test_data)["rmse"]
        result.append(error)

    return np.mean(result)
