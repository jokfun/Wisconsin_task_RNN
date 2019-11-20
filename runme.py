from createGraph import launcher

if __name__ == "__main__":
    #Hyper-parameters you can change
    ###################################

    #Number of hidden rules
    nb_dimension = 3
    #Number of possible cards
    nbrcartes = 4
    #spectral_Radius
    spectral_Radius = 1e-4
    #sparsity of the reservoir
    sparsity = 0.65
    #sparsity input
    sparsityin = 0.60
    #reward sparsity
    sparsityrw = 0.8
    #leaking rate
    leak = 0.6
    #adding noise inside the model
    noise = 1e-2
    #regularization of the model
    alpha = 1e-6
    #Number of time steps during which a stimusize lasts
    # stimusize = 5
    stimulus_length_training = 10
    #feedback scale
    fbscale = 1e-3
    #input scale
    inputscale = 1.
    #scaling of the rewardit 
    rewardscale = 5.0
    #Different sizeo of reservoir you'll test
    # sizeres = [100,500,900]
    sizeres = 100
    #Number of reservoir generated
    inst_number = 1 #5
    #Different data size you'll test
    # datasize = [i for i in range(200,2000,200)]
    #datasize = [i for i in range(20,200,20)]
    datasize = [300]
    #Number of rules which will switch during learning, 1 is the simplest task
    volatility = 15
    #Interval of volatility variation
    variability = 3
    #Number of internals neuron you want to print, will be choosen randomly
    # nbr_neur = 5
    nbr_neur = 15
    #break_time
    _stimulus_length_warm_up = 3
    #sparsity of the best choice
    sparsitychoice=0.9
    #scaling of the best choice
    choicescale=5.0
    
    stimulus_updating_reward=2

    ###################################

    import sys
    if len(sys.argv)==3:
        try:
            nb_dimension = int(sys.argv[1])
            nbrcartes = int(sys.argv[2])
        except Exception as e:
            print(e)
            exit()
    else:
        print("Default - Dimension :",nb_dimension,",Nbr Cards :",nbrcartes)
    
    launcher(nb_dimension=nb_dimension,nbrcartes=nbrcartes,spectral_Radius=spectral_Radius,sparsity=sparsity,leak=leak,noise=noise,
    alpha=alpha,stimulus_length_training=stimulus_length_training,fbscale=fbscale,inputscale=inputscale,sizeres=sizeres,inst_number=inst_number,
    datasize=datasize,volatility=volatility,nbr_neur=nbr_neur,rewardscale=rewardscale,_stimulus_length_warm_up=_stimulus_length_warm_up,
    sparsityin=sparsityin,sparsityrw=sparsityrw,sparsitychoice=sparsitychoice,choicescale=choicescale,variability=variability,stimulus_updating_reward=stimulus_updating_reward)
