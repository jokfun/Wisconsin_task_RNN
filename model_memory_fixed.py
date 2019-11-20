import tqdm
import numpy as np

class Force:
    
    def __init__(self,shape,sparsity=0.25,spectral_Radius=1.0,alpha=1.,leak=1.0,_stimulus_length_warm_up=2,
                noise=0.0,seed=0,stimulus_length_training=1,fbscale=0.5,inputscale=1.,rewardscale=1.,moving_average=30,
                sparsityin=1.0,sparsityrw=1.0,sparsitychoice=1.0,choicescale=1.,stimulus_updating_reward=0,minchoice=0.5):
                    
        """
            Force learning algorithm
            Paper : "Generating Coherent Patterns of Activity from Chaotic Neural Networks", Susillo et al.
            shape : shape of the model (intput size, reservoir size, output size)
            sparsity : sparisty of the reservoir, default : 0.25
            spectral_Radius : scaling coeffiscient, default : 1.0
            alpha : regularization coeffiscient in force learning (identity matrix), default : 1.0
            leak : leaking rate, default : 1.0
            _stimulus_length_warm_up : time during the reservoir is update without new input, default : 2
            noise : noise in the reservoir, default 0.0
            seed : seed of the model, default : 0
            stimulus_length : duration of a stimulus, default : 1
            fbscale : scaling the feedback, default : 0.5
            inputscale : scaling of the input, default : 1.0
            rewardscale : scaling of the reward, default : 1.0
            sparsityin : sparsity of the inputs, default : 1.0
            sparsityrw : sparsity of the reward, 1.0
            sparsitychoice : sparsity of the last choice of cards, default : 1.0
            choicescale : scaling of the choice, default : 1.0
            stimulus_updating_reward : time during the reward can be update, default : 1.0
        """
        #random is initiate with a specific seed
        #rng = np.random.mtrand.RandomState(seed)
        """
        self.volatility=volatility*(stimulus_length+_stimulus_length_warm_up)
        self.count_rules = 0
        self.cursor = 0
        """
        
        rng = np.random
        """
            input weight, 
            we connect it with a specific part of the reservoir, depend on the sparsity
            then we scale weights
        """
        self.W_in = rng.uniform(-1.0, 1.0, (shape[1], shape[0]))
        self.W_in[rng.uniform(0.0, 1.0, self.W_in.shape) > sparsityin] = 0.0
        self.W_in *= inputscale
        
        """
            internal weight
            internal connection of the reservoir depend on the sparsity
            then we scale weights
        """
        self.W_rc = rng.uniform(-0.5, 0.5, (shape[1], shape[1]))
        self.W_rc[rng.uniform(0.0, 1.0, self.W_rc.shape) > sparsity] = 0.0
        self.W_rc *= spectral_Radius / np.max(np.abs(np.linalg.eigvals(self.W_rc)))
        
        """
            feedback weight
            connection between the output to the reservoir
        """
        self.W_fb = rng.uniform(-1.0, 1.0, (shape[1], shape[2]))
        self.W_fb *= fbscale
        
        #output weight
        self.W_out = rng.uniform(-1.0, 1.0, (shape[2], shape[1]))
        
        #initiate all the other values
        #permanent values for the reservoir
        self.leak = leak
        self.sparsity = sparsity
        self.stimulus_length_training = stimulus_length_training
        self.noise = noise
        self.seed = seed
        self.alpha = alpha
        self._stimulus_length_warm_up=_stimulus_length_warm_up
        
        #we save all the predicted and desired output in 2 arrays
        self.savedesired = self.saveoutput = np.array([np.zeros(shape[2])])
        
        #internal values, values of the neurons inside the reservoir
        self.internals = np.zeros((shape[1]))
        
        #we will save all the internals state
        self.saveinternals = np.array([np.copy(self.internals)])
        
        #compute an initial output, depend only the internal state
        self.output = np.dot(self.W_out, self.internals)
        
        """
            reward can take values -1. or 1., initiate to 0 cause we can't know what's the first card is
            we initiate weights connection between the reward and the reservoir
        """
        self.reward = -1.
        self.W_rw = rng.uniform(-0.5, 0.5, (shape[1]))
        self.W_rw[rng.uniform(0.0, 1.0, self.W_rw.shape) > sparsityrw] = 0.0
        self.W_rw *= rewardscale
        
        #we will save all the reward in a specific array
        self.savereward = np.zeros((1))
        
        #creation of the feedback choice, and its specific connection weight
        self.choice_fb = np.zeros(shape[2])
        self.W_ch = rng.uniform(-0.5, 0.5, (shape[1], shape[2]))
        self.W_ch[rng.uniform(0.0, 1.0, self.W_ch.shape) > sparsitychoice] = 0.0
        self.W_ch*=choicescale
        
        #save sleep activity
        self.sleep_activity = np.array([0])
        
        #save of w_out
        self.saveWout = np.array([0])
        
        #save none choice
        self.nonechoice = np.array([1])
        
        #stimulus_updating_reward phase
        self.stimulus_updating_reward = int(stimulus_updating_reward)
        
        #define the min born of the choice
        self.minchoice = minchoice
        
        #value of the interval of the moving_average
        self.moving_average = moving_average
        
    def train(self,data):
        #extraction of inputs and teacher from thr data
        inputs, teacher = data["input"], data["output"]
        
        start = len(self.saveinternals)
        #initial identity matrix of the algorithm
        P = np.identity(self.internals.shape[0])/self.alpha
        
        for i in range(0,len(data)):
            for k in range(self.stimulus_length_training+self._stimulus_length_warm_up+self.stimulus_updating_reward):
                
                #print(str(self.cursor)+" "+str(self.count_rules))
                #computing the values of the reservoir
                z = (np.dot(self.W_rc,self.internals) +
                     np.dot(self.W_in,inputs[i]) +
                     np.dot(self.W_fb,self.output)+
                     np.dot(self.W_rw,self.reward)+
                     np.dot(self.W_ch,self.choice_fb) )#+
                     #np.dot(self.W_rl,self.choice_rule))
                
                #we create the noise for the reservoir
                add_noise = self.noise * np.random.uniform(-1.,1.,self.internals.shape)
                
                #compute the values with the activation function and noise added
                copintern = np.tanh(z) + add_noise
                
                #compute the internal values with the leaking rate
                copintern = (1-self.leak)*(self.internals) + self.leak*copintern

                #update of the var self.internals
                self.internals = copintern
                
                #compute the output
                z = self.internals
                self.output = np.dot(self.W_out,z)
                
                #we select the card to chose depending on a softmax with the output
                choix = self.makeChoice(self.output)
                
                if choix==None:
                    self.choice_fb.fill(0)
                    self.reward=-1
                    self.nonechoice = np.concatenate((self.nonechoice,[1]),axis=0)
                else:
                    #update of the feedback with the softmax result
                    self.choice_fb.fill(0)
                    self.choice_fb[choix] = 1.
                    self.nonechoice = np.concatenate((self.nonechoice,[0]),axis=0)
                     
                    #reward bases on the softmax and the desired value
                    if self.reward==0:
                        self.reward = 1. if choix==np.argmax(teacher[i]) else -1.
                
                if k < self.stimulus_length_training+self._stimulus_length_warm_up:
                    self.reward = 0
                
                copy_Wout = self.W_out.copy()
                
                #we enter the condition if we are not in the time break
                if k >= self._stimulus_length_warm_up and k < self.stimulus_length_training+self._stimulus_length_warm_up:                    
                    """
                        Here is the learning phase
                    """
                    #compute the error
                    e = self.output - teacher[i]
                    #we compute a P matrix with the new internal state
                    Pz = np.dot(P, z)[:,None]
                    #we update the P matrix, dependent on the last one, and the new state
                    #it's this which take so much time, cause we have to compute P-1, which is a N*N matrix
                    P -= np.dot(Pz, Pz.T)/(1+np.dot(z.T, Pz))
                    #a P matrix with the updated P and the actual state is compute
                    Pz = np.dot(P, z)
                    #update of the output eight with the P matrix
                    self.W_out -= np.dot(Pz[:,None], e[None, :]).T
                    
                    self.sleep_activity = np.concatenate((self.sleep_activity,[1]),axis=0)
                else:
                    self.sleep_activity = np.concatenate((self.sleep_activity,[0]),axis=0)
                #internals saved array is updtate
                self.saveinternals= np.concatenate((self.saveinternals,[copintern]),axis=0)
                
                #reward saved array is updtate
                self.savereward = np.concatenate((self.savereward,[self.reward]))
                
                #predicted saved array is updtate
                self.saveoutput = np.concatenate((self.saveoutput,[self.output]),axis = 0)
                
                #desired saved array is updtate
                self.savedesired = np.concatenate((self.savedesired,[teacher[i]]),axis = 0)
                
                #save the evolution of the w_out
                self.saveWout = np.concatenate((self.saveWout,[np.linalg.norm(np.absolute(self.W_out-copy_Wout))]),axis=0)
                
                
        #Prediction are compute with the internals saved array and the updated output weight
        out = np.dot(self.saveinternals[start:],self.W_out.T)
        desired = data["output"]
        
        #compute the rmse
        mse = []
        desired = np.repeat(desired, (self.stimulus_length_training+self._stimulus_length_warm_up+self.stimulus_updating_reward), axis = 0)
        """
        for i in range(len(desired)):
            for k in range(self.stimulus_length+self._stimulus_length_warm_up):
                mse.append( np.sum((out[i*(self.stimulus_length+self._stimulus_length_warm_up)+k] - desired[i] ) **2))
        """
        
        mse = np.mean(np.sum((out - desired)**2, axis = 1))
        
        rmse = np.sqrt(mse)
        
        #compute the accuracy
        accuracy =  np.count_nonzero(self.savereward[start:] == 1) / len(self.savereward[start:])
        
        #compute the cumulative advantage score
        cumulative_advantage_score = 0
        current_reward = 0
        score = 0
        for element in self.savereward[start:]:
            if element==current_reward:
                score+=1
            else:
                cumulative_advantage_score += (score**2)*current_reward
                score = 0
                current_reward=element
        
        #Creating the moving average
        
        tab_moving_average = []
        for j in range(len(self.savereward[start:])):
            if j < self.moving_average:
                tab_moving_average.append(0)
            else:
                value = np.mean(self.savereward[start+j-self.moving_average:start+j])
                tab_moving_average.append(value)
        self.tab_moving_average = np.array(tab_moving_average)
        
        return {
        "rmse" : rmse,
        "accuracy" : accuracy,
        "cumulative_advantage_score" : cumulative_advantage_score
        }
        
    def test(self,data):
        total_error = 0
        
        start = len(self.saveinternals)
        #extraction of inputs and teacher from thr data
        inputs, teacher = data["input"], data["output"]
        
        for i in range(0,len(data)):
            for k in range(self.stimulus_length_training+self._stimulus_length_warm_up+self.stimulus_updating_reward):
                
                #computing the values of the reservoir
                z = (np.dot(self.W_rc,self.internals) +
                     np.dot(self.W_in,inputs[i]) +
                     np.dot(self.W_fb,self.output)+
                     np.dot(self.W_rw,self.reward)+
                     np.dot(self.W_ch,self.choice_fb))#+
                     #np.dot(self.W_rl,self.choice_rule))
                
                #we create the noise for the reservoir
                add_noise = self.noise * np.random.uniform(-1.,1.,self.internals.shape)
                
                #compute the values with the activation function and noise added
                copintern = np.tanh(z) + add_noise
                
                #compute the internal values with the leaking rate
                copintern = (1-self.leak)*(self.internals) + self.leak*copintern
                
                #update of the var self.internals
                self.internals = copintern
                
                #compute the output
                z = self.internals
                self.output = np.dot(self.W_out,z)
                
                #we select the card to chose depending on a softmax with the output
                choix = self.makeChoice(self.output)
                
                self.choice_fb.fill(0)
                
                if choix==None:
                    self.reward=-1
                    self.nonechoice = np.concatenate((self.nonechoice,[1]),axis=0)
                else:
                    #update of the feedback with the softmax result
                    self.choice_fb[choix] = 1.
                    self.nonechoice = np.concatenate((self.nonechoice,[0]),axis=0)
                     
                    #reward bases on the softmax and the desired value
                    if self.reward==0:
                        self.reward = 1. if choix==np.argmax(teacher[i]) else -1.
                
                if k < self.stimulus_length_training+self._stimulus_length_warm_up:
                    self.reward = 0
                
                #we enter the condition if we are not in the time break
                if k >= self._stimulus_length_warm_up and k < self.stimulus_length_training+self._stimulus_length_warm_up:
                    self.sleep_activity = np.concatenate((self.sleep_activity,[1]),axis=0)
                else:
                    self.sleep_activity = np.concatenate((self.sleep_activity,[0]),axis=0)
                    
                #internal saved array is updtate
                self.saveinternals= np.concatenate((self.saveinternals,[copintern]),axis=0)
                
                #reward saved array is updtate
                self.savereward = np.concatenate((self.savereward,[self.reward]))
                
                #predicted saved array is updtate
                self.saveoutput = np.concatenate((self.saveoutput,[self.output]),axis = 0)
                
                #desired saved array is updtate
                self.savedesired = np.concatenate((self.savedesired,[data["output"][i]]),axis = 0)
        
        #Prediction are compute with the internals saved array and the updated output weight
        out = np.dot(self.saveinternals[start:],self.W_out.T)
        desired = data["output"]
        
        #compute the rmse
        mse = []
        desired = np.repeat(desired, (self.stimulus_length_training+self._stimulus_length_warm_up+self.stimulus_updating_reward), axis = 0)
        """
        for i in range(len(desired)):
            for k in range(self.stimulus_length+self._stimulus_length_warm_up):
                mse.append( np.sum((out[i*(self.stimulus_length+self._stimulus_length_warm_up)+k] - desired[i] ) **2))
        """
        
        mse = np.mean(np.sum((out - desired)**2, axis = 1))
        
        rmse = np.sqrt(mse)
        
        #compute the accuracy
        accuracy =  np.count_nonzero(self.savereward[start:] == 1) / len(self.savereward[start:])
        
        #compute the cumulative advantage score
        cumulative_advantage_score = 0
        current_reward = 0
        score = 0
        for element in self.savereward[start:]:
            if element==current_reward:
                score+=element
                cumulative_advantage_score+=score
            else:
                score = 0
                cumulative_advantage_score+=element
                current_reward=element
        
        return {
        "rmse" : rmse,
        "accuracy" : accuracy,
        "cumulative_advantage_score" : cumulative_advantage_score
        }
    
    def makeChoice(self,tab):
        """
            Just use the argmax
        """
        choice = np.argmax(tab)
        if tab[choice]<self.minchoice:
            return None
        else:
            return choice
        """
            Softmax function : will return the choosen card depend on the proba
            tab : an array of real values, must have no 0 values
        """
        """
        tab+=min(tab)+1e-8
        softmax = lambda z:abs(z)/np.sum(abs(z))
        soft_value = softmax(tab)
        choice = np.random.choice(len(soft_value), 1, p=soft_value)
        return choice[0]
        """

    def checkfixed(self):
        pass
