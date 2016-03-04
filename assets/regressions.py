from sklearn import linear_model
import numpy as np

#y's are between 0 and 1 if logistic, continuous if linear
#model_type = 'linear' or 'logistic'
#cv_score is classification accuracy if logistic, R^2 if linear
def regression(X_train,X_val,X_test,y_train,y_val,y_test,reg_min=15,reg_max=15,reg_dist=2,model_type='linear'):
    reg_range = np.exp(np.array(xrange(-reg_min,reg_max))*reg_dist)

    best_model = None
    best_score = 0

    for reg in reg_range:
        if(model_type == 'linear'):
            model = linear_model.Lasso(alpha=reg)
        elif(model_type == 'logistic'):
            model = linear_model.LogisticRegression(penalty='l1',C=1/reg)
        else:
            print "model type not valid"
            return
        model.fit(X_train, y_train)
        score = model.score(X_val,y_val)
        if(score > best_score):
            best_score = score
            best_model = model

    cv_score = best_model.score(X_test,y_test)
    return best_model, cv_score, best_score

