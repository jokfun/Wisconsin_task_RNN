# 1 seul reward par stimulus (la reward n'est calculé qu'une seule fois par chaque stimulus)
MODEL_FIXED = True
from model_memory_fixed import Force

# # la reward est calculé/updaté à chaque fois qu'un choix est fait (à tous les pas de temps)
# # équivalent à avoir le droit à plusieurs essais (choix de carte) pour chaque stimulus
# from model_memory import Force
# MODEL_FIXED = False

from time import time
import gener as gen
import numpy as np
from random import randint,shuffle

def launcher(nb_dimension=3,
        nbrcartes= 4,
        spectral_Radius = 0.7,
        sparsity = 0.8 ,
        leak = 0.9,
        noise = 0,
        alpha = 1e-5,
        stimulus_length_training = 3,
        fbscale = 1.,
        inputscale = 0.5,
        sizeres = 300,
        inst_number = 10,
        datasize = [20,40,60,80,100],
        volatility = 1,
        nbr_neur = 15,
        rewardscale=1.,
        _stimulus_length_warm_up=2,
        sparsityin=1.0,
        sparsityrw=1.0,
        sparsitychoice=0.9,
        choicescale=1.0,
        variability = 3,
        stimulus_updating_reward=0):

    # Depend on the type of Force imported, we need update how are display the graph
    ## xav: "hack pour éviter qu'on change le pas de temps dans les graphs quand on est en mode 'fixed' "
    addForGraph = 0
    try:
        Force.checkfixed
        addForGraph = stimulus_updating_reward
    except Exception as e:
        pass

    save_plot = True

    #Compare test error (accuracy) with this value in order to find best model
    bestscore = -1

    #array of test errors
    result = []

    #array of learning errors
    resultappr = []


    #local array for test erros, for a specific reservoir size
    add = []

    #local array for learning erros, for a specific reservoir size
    addappr = []

    size_input = ((nbrcartes + nbrcartes**2)*nb_dimension)
    size_input = nbrcartes * nb_dimension
    label_size = nbrcartes #+ nb_dimension
    features = []
    labels = []


    for k in datasize:
        print("\n\n===============================")
        print("Reservoir size :",sizeres)
        print("Data size :",k)
        debut = time()

        start = time()

        value = 0

        lsrule = []

        rd_variability = randint(-variability,variability)
        #Creation of features and labels
        while len(features)<k:
            if len(features) % (volatility+rd_variability) == 0:
                value = randint(0,nb_dimension-1)
                rd_variability = randint(-variability,variability)
            feat,label = gen.carte(nb_dimension,nbrcartes,value)
            #Will be used for the graph
            lsrule.append(value)
            labels.append(label)
            features.append(feat)

        lsrule = np.array(lsrule)
        lsrule = np.repeat(lsrule,stimulus_length_training+_stimulus_length_warm_up+addForGraph,axis=0)


        data = np.zeros(k, dtype = [ ("input",  float, (size_input,)),
                                     ("output", float, (label_size,))])
        data["input"] = features[:k]
        data["output"] = labels[:k]

        #Split the data in training data and testing data
        train_data = data[:int(2/3*len(data))]
        test_data = data[int(len(data)*2/3):]


        for res in range(inst_number):
            """
            model = mo.generate_model(shape=(size_input,sizeres,label_size),
            spectral_Radius=spectral_Radius,sparsity=sparsity,leak=leak,noise=noise,steptime=stimulus_length)
            """

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

            print("Model creation :",time()-start,"sec")

            error = model.test(test_data)["rmse"]
            print("Test accuracy : {0}".format(error))


            #learning phase
            error = model.train(train_data)["rmse"]
            print("Training accuracy : {0}".format(error))
            addappr.append(error)

            #test model
            error = model.test(test_data)["rmse"]
            # error = model.test(test_data)["cumulative_advantage_score"]
            print("Test accuracy : {0}".format(error))
            add.append(error)

            #TODO CAN BE CHANGED: xav: error param available
            # return {
            # "rmse" : rmse,
            # "accuracy" : accuracy,
            # "cumulative_advantage_score" : cumulative_advantage_score
            # }

        resultappr.append(np.mean(addappr))
        result.append(np.mean(add))
        add = []
        addappr = []

    print("Erreur Moyenne :",np.mean(result))
    print(model.sleep_activity.shape,lsrule.shape)
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.colors as pltc

    matplotlib.rcParams.update({'font.size': 13})

    # general info for users
    stimulus_length = stimulus_length_training + _stimulus_length_warm_up+stimulus_updating_reward
    total_nr_timesteps = stimulus_length * datasize[0]

    ###################################################################################
    #Graph to show accuracy evolution

    f = plt.figure(figsize=(20, 6))
    f.suptitle("Mean of the evolution of the accuracy \nnb_dimension : "+str(nb_dimension) + ", Number of choice : "+str(nbrcartes))
    plt.plot(datasize,resultappr,'r',label="Learning")
    plt.plot(datasize,result,'b',label="Test")
    plt.plot(datasize,resultappr,'ro')
    plt.plot(datasize,result,'bo')
    plt.xlabel('Data size')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    plt.ylim(min(min(resultappr),min(result)),max(max(resultappr),max(result)))

    if save_plot:
        f.savefig("example/result_dim_"+str(nb_dimension)+"_res_"+str(nbrcartes)+".png")

    #alldata = np.arange(len(model.saveinternals))

    ##################################################################################
    #Graph to show the feed back evolution

    f = plt.figure(figsize=(20, 6))
    f.suptitle("Reward Evolution")
    plt.plot(model.savereward-2,"r",label="reward")
    plt.axvline(x=len(test_data)*stimulus_length_training,linewidth=0.8)
    plt.axvline(x=(len(train_data)+len(test_data))*stimulus_length_training,linewidth=0.8)
    plt.yticks([])
    plt.text(10, 10, "Random phase", fontsize=12)
    #plt.text(0, , r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)

    if save_plot:
        f.savefig("example/Feedback_Evolution_result_dim_"+str(nb_dimension)+"_res_"+str(nbrcartes)+".png")


    ##################################################################################
    #Graph for internals activity

    all_colors =[k for k,v in pltc.cnames.items()]
    nbr_neur = 15
    tab = [i for i in range(len(model.saveinternals[:][0]))]
    shuffle(tab)
    f = plt.figure(figsize=(30,15))
    f.subplots_adjust(hspace = 0.6 )
    f.suptitle("Internals evolution")

    ax = f.add_subplot(3,1,1)
    for i in range(nbr_neur):
        ax.plot(model.saveinternals[:len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph),tab[i]])
    ax.set_title("Random phase")
    ax.set_ylim([-1.1,1.1])

    start = len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)
    ax = f.add_subplot(3,1,2)
    for i in range(nbr_neur):
        ax.plot(model.saveinternals[start:start+len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph),tab[i]])
    ax.set_title("Training phase")
    ax.set_ylim([-1.1,1.1])

    start+= len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)
    ax = f.add_subplot(3,1,3)
    for i in range(nbr_neur):
        ax.plot(model.saveinternals[start:,tab[i]])
    ax.set_title("Test phase")
    ax.set_ylim([-1.1,1.1])

    if save_plot:
        f.savefig("example/Internals_Evolution_result_dim_"+str(nb_dimension)+"_res_"+str(nbrcartes)+".png")

    ##################################################################################
    #Graphs to compare the 3 phases

    f = plt.figure(figsize=(25,20))
    f.suptitle("Random phase\nLength of a stimulus :"+str(stimulus_length)+"\nTotal of time step :"+str(total_nr_timesteps))
    f.subplots_adjust(hspace = 0.6 )
    outsize = len(model.saveoutput[0])+5
    for i in range(outsize-5):
        ax = f.add_subplot(outsize,1,i+1)
        ax.plot(model.saveoutput[:len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph),i],'r',label="predict")
        ax.plot(model.savedesired[:len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph),i],"b",label="desired")
        ax.set_title("Desired and Predicted output - Card "+str(i))
        ax.legend(loc=4)
        ax.set_ylim([-0.5,1.5])


    ax = f.add_subplot(outsize,1,outsize-4)
    ax.plot(model.savereward[:len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up)+addForGraph],"r")
    ax.set_title("Reward t")
    ax.set_ylim([-1.1,1.1])

    ax = f.add_subplot(outsize,1,outsize-3)
    ax.plot([0]+model.savereward[:-1+len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)],"r")
    ax.set_title("Reward t-1")
    ax.set_ylim([-1.1,1.1])

    ax = f.add_subplot(outsize,1,outsize-2)
    ax.plot(lsrule[int(2/3*len(lsrule)):],"r")
    ax.set_title("Evolution règle")
    ax.set_ylim([-0.1,3.1])

    ax = f.add_subplot(outsize,1,outsize-1)
    ax.plot(model.sleep_activity[:len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)],"g")
    ax.set_title("Stimulus activity")
    ax.set_ylim([-0.1,1.1])

    ax = f.add_subplot(outsize,1,outsize)
    ax.plot(model.nonechoice[:len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)],"b")
    ax.set_title("No card choice")
    ax.set_ylim([-0.1,1.1])

    if save_plot:
        f.savefig("example/Random_Phase_result_dim_"+str(nb_dimension)+"_res_"+str(nbrcartes)+".png")

    start = len(test_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)

    f = plt.figure(figsize=(20,15))
    f.suptitle("Learning phase\nLength of a stimulus :"+str(stimulus_length)+"\nTotal of time step :"+str(total_nr_timesteps))
    f.subplots_adjust(hspace = 0.6 )
    outsize = len(model.saveoutput[0])+7
    for i in range(outsize-7):
        ax = f.add_subplot(outsize,1,i+1)
        ax.plot(model.saveoutput[start:start+len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph),i],'r',label="predict")
        ax.plot(model.savedesired[start:start+len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph),i],"b",label="desired")
        ax.set_title("Desired and Predicted output - Card "+str(i))
        ax.legend(loc=4)
        ax.set_ylim([-0.5,1.5])

    ax = f.add_subplot(outsize,1,outsize-6)
    ax.plot(model.tab_moving_average)
    ax.set_title("Moving Average")

    print(len(model.tab_moving_average),len(model.savereward[start:start+len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)]))
    ax = f.add_subplot(outsize,1,outsize-5)
    ax.plot(model.savereward[start:start+len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)],"r")
    ax.set_title("Reward t")
    ax.set_ylim([-1.1,1.1])

    ax = f.add_subplot(outsize,1,outsize-4)
    ax.plot(model.savereward[start-1:-1+start+len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)],"r")
    ax.set_title("Reward t-1")
    ax.set_ylim([-1.1,1.1])

    ax = f.add_subplot(outsize,1,outsize-3)
    ax.plot(lsrule[:int(2/3*len(lsrule))],"r")
    ax.set_title("Evolution règle")
    ax.set_ylim([-0.1,3.1])

    ax = f.add_subplot(outsize,1,outsize-2)
    ax.plot(model.sleep_activity[start:start+len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)],"g")
    ax.set_title("Stimulus activity")
    ax.set_ylim([-0.1,1.1])

    ax = f.add_subplot(outsize,1,outsize-1)
    ax.plot(model.saveWout)
    ax.set_title("W_out evolution")

    ax = f.add_subplot(outsize,1,outsize)
    ax.plot(model.nonechoice[start:start+len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)],"b")
    ax.set_title("No card choice")
    ax.set_ylim([-0.1,1.1])


    if save_plot:
        f.savefig("example/Learning_Phase_result_dim_"+str(nb_dimension)+"_res_"+str(nbrcartes)+".png")

    start+=len(train_data)*(model.stimulus_length_training+_stimulus_length_warm_up+addForGraph)

    f = plt.figure(figsize=(20,15))
    f.suptitle("Testing phase\nLength of a stimulus :"+str(stimulus_length)+"\nTotal of time step :"+str(total_nr_timesteps))
    f.subplots_adjust(hspace = 0.6 )
    outsize = len(model.saveoutput[0])+5
    for i in range(outsize-5):
        ax = f.add_subplot(outsize,1,i+1)
        ax.plot(model.saveoutput[start:,i],'r',label="predict")
        ax.plot(model.savedesired[start:,i],"b",label="desired")
        ax.set_title("Desired and Predicted output - Card "+str(i))
        ax.legend(loc=4)
        ax.set_ylim([-0.5,1.5])

    ax = f.add_subplot(outsize,1,outsize-4)
    ax.plot(model.savereward[start:],"r")
    ax.set_title("Reward t (computed for input t)")
    ax.set_ylim([-1.1,1.1])

    ax = f.add_subplot(outsize,1,outsize-3)
    ax.plot(model.savereward[start-1:-2],"r")
    ax.set_title("Reward t-1 (inputed in reservoir)")
    ax.set_ylim([-1.1,1.1])

    ax = f.add_subplot(outsize,1,outsize-2)
    ax.plot(lsrule[int(2/3*len(lsrule)):],"r")
    ax.set_title("Evolution règle")
    ax.set_ylim([-0.1,3.1])

    ax = f.add_subplot(outsize,1,outsize-1)
    ax.plot(model.sleep_activity[start:],"g")
    ax.set_title("Stimulus activity")
    ax.set_ylim([-0.1,1.1])

    ax = f.add_subplot(outsize,1,outsize)
    ax.plot(model.nonechoice[start:],"b")
    ax.set_title("No card choice")
    ax.set_ylim([-0.1,1.1])

    if save_plot:
        f.savefig("example/Testing_Phase_result_dim_"+str(nb_dimension)+"_res_"+str(nbrcartes)+".png")

    ##################################################################################

    if not save_plot:
        plt.show()

    if save_plot:
        with open("example/hyperparameters__dim_"+str(nb_dimension)+"_res_"+str(nbrcartes)+".txt","w") as f:
            f.write("nb_dimension = "+str(nb_dimension)+"    #Nombre de dimension du jeu\n")
            f.write("nbrcartes = "+str(nbrcartes)+"    #Nombre de carte possible sur le plateau en dehors de la carte de test\n")
            f.write("spectral_Radius = "+str(spectral_Radius)+"    #Coeffiscient de scaling\n")
            f.write("sparsity = "+str(sparsity)+"    #Sparsité à l'intérieur du reservoir\n")
            f.write("leak = "+str(leak)+"    #leaking rate, influe la dynamique du reservoir\n")
            f.write("alpha = "+str(alpha)+"    #Coeffiscient de régularisation dans le force learning (matrice d'identité)\n")
            f.write("stimulus_length_training = "+str(stimulus_length_training)+"    #Durée d'apprentissage d'un input\n")
            f.write("fbscale = "+str(fbscale)+"    #Scaling du feedback\n")
            f.write("inputscale = "+str(inputscale)+"    #Scaling de l'input\n")
            f.write("rewardscale = "+str(rewardscale)+"    #Scaling du reward\n")
            f.write("noise = "+str(noise)+"      #Bruit ajouté aux sorties du reservoir\n")
            f.write("sizeres = "+str(sizeres)+"    #Taille du reservoir\n")
            f.write("inst_number = "+str(inst_number)+"    #Nombre d'instance réalisée (créer une moyenne plus pertinente)\n")
            f.write("datasize = "+str(datasize)+"    #Taille du jeu de donnée\n")
            f.write("volatility = "+str(volatility)+"    #mesure statistique de la dispersion, ici durée d'utilisation d'une règle dans le jeu de donnée\n")
            f.write("variability = "+str(variability)+"    #Intervalle de variation de la volatilité\n")
            f.write("nbr_neur = "+str(nbr_neur)+"    #Nombre de neurone présentés dans le graphique de l'activité du reservoir\n")
            f.write("_stimulus_length_warm_up = "+str(_stimulus_length_warm_up)+"    #Durée de mise à jour du reservoir, où il tourne dans le vide\n")
            f.write("sparsityin = "+str(sparsityin)+"    #Sparsité de l'input\n")
            f.write("choicescale = "+str(choicescale)+"    #Scaling du choix\n")
            f.write("sparsitychoice = "+str(sparsitychoice)+"    #Sparsité du choix\n")
            f.write("sparsityrw = "+str(sparsityrw)+"    #Sparsité du reward\n")
            if not addForGraph == 0 :
                f.write("stimulus_updating_reward = "+str(stimulus_updating_reward)+"    #Phase durant laquelle le reward est update\n")
            f.close()
