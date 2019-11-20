# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Created on Thu Jun  4 13:00:56 2015

@author: hinaut #/at\# informatik.uni-hamburg.de

Some interesting comments on Hyperopt:
http://stackoverflow.com/questions/24673739/hyperopt-set-timeouts-and-modify-space-during-execution
"""

# import action_perf_naive_exp_cv
import experiment


# import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval, rand
from hyperopt.pyll.stochastic import sample

import pprint
import numpy as np
import json
import copy

# import git
import pickle

general_param = None
#SAVE_DIR = "../RES_TEMP_HP/"
SAVE_DIR = "final/"
#MULTIBALL = True #use the 3 corpora (FR, EN, FREN) at once in the parameter search
MULTIBALL = False #use the 3 corpora (FR, EN, FREN) at once in the parameter search
#ERR_and_3STD = False
ERR_and_3STD = False
#ERR_and_time = True
ERR_and_time = False

curr_eval = 0
SAVE_TRIAL_OBJ_EVERY = 5

def byteify(input):
    """
    There's no built-in option to make the json module functions return byte
    strings instead of unicode strings. However, this short and simple recursive
    function will convert any decoded JSON object from using unicode strings to
    UTF-8-encoded byte strings
    Just call this on the output you get from a json.load or json.loads call.

    http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python#13105359
    """
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key,value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def geht_los_einfach(mean, std, run_time): #get loss simple
    return mean, None

def geht_los(mean, std, run_time): #get loss
    if ERR_and_time:
        time_loss = np.log(run_time/60. + 1)
    else:
        time_loss = None
        if ERR_and_3STD and ERR_and_time:
            #            loss_tmp = (mean+3*std)*(np.log(1+run_time)/3600.)
            loss_tmp = (mean + 3*std) * time_loss
        elif ERR_and_3STD:
            loss_tmp = mean + 3*std
        elif ERR_and_time:
            loss_tmp = mean * time_loss
        else:
            loss_tmp = mean
            return loss_tmp, time_loss




def lookup_search_space():
    # lookup_search_space: Plot the search space given instead of launching a parameter search.
    import pylab as pl
#    search_space = [hp.loguniform('sr', -2.3, 2.3),
##                    hp.qloguniform('iss', -2.3, 2.3,0.2),
#                    hp.quniform('iss', 0.1, 10,0.2),
##                    hp.randint('N', 100),
#                    hp.quniform('N', 100, 300, 100),
##                    hp.quniform('leak', 0.01, 1.0,0.1),
##                    hp.qloguniform('leak', -4.5, 0,0.01)]
##                    hp.qloguniform('leak', -4.5, 0,0.01)]
#                    hp.loguniform('leak', -4.5, 0,0.01)]

    # search_space = [hp.quniform('N', 500, 501, 500),
    #                 hp.uniform('sr',0.01, 10.),
    #                 hp.uniform('iss',0.01, 10.),
    #                 hp.uniform('leak', 0.006737946999085467, 1.0)]
    search_space = [
                    hp.quniform('sr', 1, 1, 1),
                    hp.loguniform('iss',np.log(0.001), np.log(10)),
                    # hp.quniform('N', 20, 21, 20),
                    # hp.quniform('fb_proba', 0.0, 1.0, 0.2), #hp.quniform('fb_proba', 0.1, 1.0, 0.15)
                    # hp.quniform('fb_proba', 0.0, 1.0, 0.5), #hp.quniform('fb_proba', 0.1, 1.0, 0.15)
                    # hp.quniform('fb_proba', 0, 15,3), #hp.quniform('fb_proba', 0.1, 1.0, 0.15)
                    # hp.quniform('fb_proba',  0.05, 1.0, 0.05), #hp.quniform('fb_proba', 0.1, 1.0, 0.15)
                    hp.quniform('wash_nr_time_step', 0, 4,1), #hp.quniform('fb_proba', 0.1, 1.0, 0.15)
                    # hp.qloguniform('fb_proba', np.log(0.01), np.log(1.), 1.), #hp.quniform('fb_proba', 0.1, 1.0, 0.15)
                    hp.loguniform('leak', -10, 0.)]

    nr_samples = 1000

    print ("search_space", search_space)
#    N_s = [sample(search_space)[2] for _ in range(100)]

    # look at SR distrubution
    N_s = [sample(search_space)[0] for _ in range(nr_samples)]
    N_s.sort()
    pl.figure()
    pl.plot(range(len(N_s)), N_s)
    pl.title('spectral radius')

    # look at ISS distrubution
    N_s = [sample(search_space)[1] for _ in range(nr_samples)]
    N_s.sort()
    pl.figure()
    pl.plot(range(len(N_s)), N_s)
    pl.title('input scaling')

    # look at number of neurons distrubution
    N_s = [sample(search_space)[2] for _ in range(nr_samples)]
    N_s.sort()
    pl.figure()
    pl.plot(range(len(N_s)), N_s)
    pl.title('number of neurons')

    # look at leak rate distrubution
    N_s = [sample(search_space)[3] for _ in range(nr_samples)]
    N_s.sort()
    pl.figure()
    pl.plot(range(len(N_s)), N_s)
    pl.title('leak')

    pl.show()



def save(o, path):
    with open(path, 'wb') as f:
        pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)




def objective_dic(params):
    """
    Objective function that takes a dictionary as argument.
    This is the function the optimizer will try to minimize by sampling parameters that are given in input.
    """

    global SAVE_TRIAL_OBJ_EVERY
    global curr_eval
    curr_eval += 1

    ### saving the trial data every SAVE_TRIAL_OBJ_EVERY
    # we save the trail at the beginning because we cannot save the results of
    #   the current trial during the call of the method we are in.
    if (curr_eval-1)%SAVE_TRIAL_OBJ_EVERY == 0:
        try:
            save(trials, SAVE_DIR+"hyperopt_trials_eval"+str(curr_eval-1)+".pkl")
            # save([t for i, t in enumerate(trials.trials) if i < len(trials.trials) ], SAVE_DIR+"hyperopt_trials_eval"+str(curr_eval)+".pkl")
        except:
            print ("!!! WARNING: COULD NOT SAVE THE PARTIAL TRIAL OBJECT.")
    
    """
    print ()
    print ("***      HP: params given to objective function:", params)
    print ("         current eval : ", curr_eval)
    print ()
    """


    current_params = {}
    current_params.update(general_param)
    current_params.update(params)

    seed = int(time.time()*10**6)
    current_params.update({'seed': seed})
    
    """
    print ("HP: all current params:", current_params)
    print ()
    """
    
    """ INIT EXPERIMENT"""
    """
    compexp = experiment.Experiment(current_params)
    compexp.launch_exp()
    """

    start_time = time.time()
    try:
        """ LAUNCH EXPERIMENT & RETRIEVE MEAN AND STD OF CROSS_VALIDATION ERROR """
        mean = experiment.Experiment(current_params)
        std = None
        #TODOO 
        end_time = time.time()
        run_time = end_time-start_time
        loss_tmp, time_loss = geht_los_einfach(mean, std,run_time) #geht_los(mean, std, run_time)
        returned_dic = {'loss': loss_tmp,
                    'status': STATUS_OK,
                    # -- store other results like this
                    'eval_time': time.time(),
                    'true_loss': mean,
                    # 'corpus_info': {'ccw_from_train_txt': ccw_from_train_txt,
                    #                    'real_l_ccw': real_l_ccw,
                    #                    'replaced_words': replaced_words},
                    'start_time': start_time,
                    'end_time': end_time,
                    'run_time': run_time,
                    'time_loss': time_loss,

                    # For info
    #                    'true_loss': type float if you pre-compute a test error for a validation error loss, store it here so that Hyperopt plotting routines can find it,
    #                    'true_loss_variance': type float variance in test error estimator,
    #                    'other_stuff': {'list_args': list_args,
    #                                    'general_param': general_param},
    #                    # -- attachments are handled differently
    #                    'attachments':
    #                        {'time_module': pickle.dumps(time.time)}
                        }
    except Exception as e:
        mean, std =  None, None
        print ("error", str(e))
        returned_dic = {'status': STATUS_FAIL,
                      'loss': 1.0, #debug: probably useless
                      'exception': str(e),
                    # -- store other results like this
                        'eval_time': time.time(),
#                        'other_stuff': {'list_args': list_args,
#                                        'general_param': general_param},
#                        # -- attachments are handled differently
#                        'attachments':
#                            {'time_module': pifckle.dumps(time.time)}
                            }

    ### SAVE FILE OF PAST SIMULATION
    try:
        if mean is not None:
            json_filename = 'err%.3f_'%mean+'hyperopt_results_1call_s'+str(seed)
        else:
            json_filename = 'exception_error__'+'hyperopt_results_1call_s'+str(seed)
        json_dic = {'returned_dic': returned_dic,
                    'current_params': current_params}

        with open(SAVE_DIR+json_filename+'.json','w') as fout:
            # json.dump(json_dic, fout)
            json.dump(json_dic, fout, separators=(',', ':'), sort_keys=True, indent=4)
    except:
        print ("WARNING: Results of current simulation were NOT saved correctly to JSON file.")
    # and test if results were saved correctly
    try:
        with open(SAVE_DIR+json_filename+'.json','r') as fin:
            data = json.load(fin)
            #pprint.pprint(byteify(data)) # in order to remove "u" in all keys of dictionary
        #print ("### Results of current simulation were saved correctly to JSON file. ###")
    except:
        print ("WARNING: Results of current simulation were NOT saved correctly to JSON file.")
    ###

    #print ("returned_dic:", returned_dic)

    return returned_dic







def main_hp_search_with_dic(simulation_name, params, sim_type=None):
    """
    Main method that is called by the __main__ method when executed.
    It is setting all things necessary for the parameter search and saving the results.
    """

    # setting the parameters as a global variable, so objective_dic() can access it
    global general_param
    general_param = params
    print ("HP: general_param", general_param)

    global SAVE_DIR
    """ Create a unique folder for the experiment with ID value (i.e. time stamp) """
    SAVE_DIR += time.strftime("%Y-%m-%d_%Hh%M__")+simulation_name+str(int(time.time()*10**6))+"/"
    import os
    os.mkdir(SAVE_DIR)

    # TODO: remove these 2 lines because they are probably useless
    # global curr_eval
    # curr_eval = 0

    # save parameters with simulation name before launching simulation
    # with open(SAVE_DIR+args.filename+'_HP-SEARCH_'+args.simulationname+'_START.json', 'w') as f:
    with open(SAVE_DIR+args.simulationname+'_START.json', 'w') as f:
        # json.dump(params,f)
        json.dump(params, f, separators=(',', ':'), sort_keys=True, indent=4)

    global trials
    trials = Trials()
    
    #EXPLORE PARAMETERS
    
    search_space = {
    "_stimulus_length_warm_up":hp.uniform('_stimulus_length_warm_up', 1,20),
    "stimulus_length_training":hp.uniform('stimulus_length_training', 1,20),
    "stimulus_updating_reward":hp.uniform('stimulus_updating_reward', 0,20)
    }
    
    """
    search_space = {
    'leak':hp.uniform('leak', 0.1,0.9),
    'alpha':hp.loguniform('alpha', np.log(1e-12), np.log(1e-1)),
    'fbscale':hp.loguniform('fbscale', np.log(1e-8), np.log(10.)),
    'inputscale':hp.loguniform('inputscale', np.log(0.999),np.log(1.)),
    'rewardscale':hp.loguniform('rewardscale', np.log(1e-8), np.log(10.)),
    'spectral_Radius':hp.loguniform('spectral_Radius', np.log(1e-8), np.log(10.)),
    'sparsity':hp.uniform('sparsity', 5e-1,1.),
    'sparsityin':hp.uniform('sparsityin', 5e-1,1.),
    'sparsityrw':hp.uniform('sparsityrw', 5e-1,1.),
    'sparsitychoice':hp.uniform('sparsitychoice', 5e-1,1.),
    'choicescale':hp.loguniform('choicescale', np.log(1e-8), np.log(10.))
    }
    """


    from functools import partial

    best = fmin(objective_dic,
            space=search_space,
            algo=partial(tpe.suggest, n_startup_jobs=100), #TPE with 'n_startup_jobs' initial random exploration
            #algo=partial(rand.suggest, n_startup_jobs=100), #RANDOM EXPLORATION
            max_evals=400,
            trials=trials)

    """ Show results with best """
    print ("*** trials ***")
    pprint.pprint(trials.trials)
    print ("*** trials[0]['result'] ***")
    pprint.pprint(trials.trials[0]['result'])
    print ("space_eval", space_eval(search_space, best))
    print ("best", best)


    """ Save results with particular format for hp_analyse_error.py """
    all_loss = [t['result']['loss'] for t in trials.trials]
    all_result = [t['result'] for t in trials.trials]
    all_vals = [t['misc']['vals'] for t in trials.trials]
#    all_res = [t['misc']['vals'].update({'loss':t['result']['loss']}) for t in trials.trials]
    tup_result_vals = [(t['result'], t['misc']['vals']) for t in trials.trials] #for saving useful and seriazable data in a JSON file
    print ("all_result")
    pprint.pprint(all_result)
    print ("all_values")
    pprint.pprint(all_vals)
    print ("max(all_loss)", max(all_loss))
    print ("min(all_loss)", min(all_loss))

    # save results to JSON file
    json_dic = {'best': best,
                'general_param':general_param,
                'tuples_result_vals': tup_result_vals}
#    with open('hyperopt_results_ter_iss075.json','w') as fout:
    with open(SAVE_DIR+'hyperopt_results_demo.json','w') as fout:
        # json.dump(json_dic, fout)
        json.dump(json_dic, fout, separators=(',', ':'), sort_keys=True, indent=4)
    # and test if results were saved correctly
    try:
#        with open('hyperopt_results_ter_iss075.json','r') as fin:
        with open(SAVE_DIR+'hyperopt_results_demo.json','r') as fin:
            data = json.load(fin)
            pprint.pprint(byteify(data)) # in order to remove "u" in all keys of dictionary
        print ("### Results were saved correctly to JSON file. ###")
    except:
        print ("WARNING: Results were NOT saved correctly to JSON file.")


    save(trials, SAVE_DIR+"hyperopt_trials.pkl")
    save(best, SAVE_DIR+"hyperopt_best.pkl")

    test_open_saved_pickle(SAVE_DIR+"hyperopt_trials.pkl")
    test_open_saved_pickle(SAVE_DIR+"hyperopt_best.pkl")

    # save parameters with simulation name after launching simulation
    with open(SAVE_DIR+args.filename+'_HP-SEARCH_'+args.simulationname+'_END.json', 'w') as f:
        # json.dump(params,f)
        json.dump(params, f, separators=(',', ':'), sort_keys=True, indent=4)


    #ring bell to say it's finished
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (1, 1000))


def test_open_saved_pickle(fname):
    try:
        trials = load(fname)
        print ("SUCCESSFULY OPENED THE PICKLE FILE")
    except:
        raise Exception( "FAIL TO OPEN PICKLE FILE")



if __name__=='__main__':
    """
    Examples of how to run the hyper-parameter search
    > python hp_experiment.py -f your-parameter-file.json -s the-simulation-name
    > python hp_experiment.py -f EN_HP_default_params_sent_comprehension.json -s titoti -t hot

    Example how to just look at how the parameters are explored
    > python hp_experiment.py -l
    """
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, help="json file containing parameters for experiment (root directory is assumed to be params/)")
    parser.add_argument("-s", "--simulationname", type=str, help="simulation_name: this will be use for the file and saving directory)")
    parser.add_argument("-l", "--lookup", action='store_true', help="lookup_search_space: Plot the search space given instead of launching a parameter search.")
    parser.add_argument("-t", "--sim_type", type=str, help="(optinal) indicates which simulation type should be chosen (for instance PHONEME or HOT WORD VEC)")
    args = parser.parse_args()

    if not args.lookup:
        with open("params/"+args.filename, 'r') as f:
            params = json.load(f)

        # save parameters with simulation name before launching simulation
        with open("params_hp/"+args.simulationname+'_START.json', 'w') as f:
            json.dump(params, f, separators=(',', ':'), sort_keys=True, indent=4)

        # launch main code
        main_hp_search_with_dic(args.simulationname, params, args.sim_type)

        # save parameters with simulation name after launching simulation
        with open("params_hp/"+args.filename+'_HP-SEARCH_'+args.simulationname+'_END.json', 'w') as f:
            # json.dump(params,f)
            json.dump(params, f, separators=(',', ':'), sort_keys=True, indent=4)

    else:
        lookup_search_space()
