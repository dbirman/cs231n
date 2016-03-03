
#author: steeve laquitaine
#purpose: save and load neural net models and their weights

import os
from assets.keras.keras.models import model_from_json
import time

def savemodel(model,modelid):

    t = time.time()
    json_string = model.to_json()
    
    #set path and check if exists
    fpa = os.getcwd()+'/assets/model/'    
    if not os.path.exists(fpa):
        os.makedirs(fpa)

    #save model    
    fpa = os.getcwd()+'/assets/model/mymodel'+str(modelid)+'.json'    
    open(fpa, 'w').write(json_string)
    
    fpw = os.getcwd()+'/assets/model/mymodelweights'+str(modelid)+'.h5'    
    model.save_weights(fpw)

    print 'done'
    elapsed = time.time() - t
    print elapsed,'sec'
    
    
def loadmodel(modelid):    
    
    t = time.time()
    
    fpa = os.getcwd()    
    #set path and check if exists
    fpa = os.getcwd()+'/assets/model/' 
    #if path does not exist
    if not os.path.exists(fpa):
        print fpa,'and saved model do not exist'
    else:
        #load model and weights
        fpa = os.getcwd()+'/assets/model/mymodel'+str(modelid)+'.json'
        fpw = os.getcwd()+'/assets/model/mymodelweights'+str(modelid)+'.h5' 
        model = model_from_json(open(fpa).read())
        model.load_weights(fpw)    
        return model
    
    print 'done'
    elapsed = time.time() - t
    print elapsed,'sec'