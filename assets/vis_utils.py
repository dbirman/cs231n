import matplotlib.pyplot as plt
import matplotlib.animation as anim 
import numpy as np
from assets.keras.keras import backend as K
from assets.keras.keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from assets.keras.keras.models import Sequential
from assets.keras.keras.layers.normalization import BatchNormalization
from assets.keras.keras.optimizers import SGD, RMSprop, Adam

def visualize_grid(W, ubound=1, padding=1):
    padding = 1
    n_feats = W.shape[0]
    n_inputs = W.shape[1]
    W_full = np.zeros((W.shape[2],W.shape[3]*n_feats + (n_feats-1)*padding,W.shape[4]*n_inputs + (n_inputs-1)*padding))
    for i in range(n_feats):
        for j in range(n_inputs):
            W_full[:,i*(padding+W.shape[3]):(i+1)*(padding+W.shape[3])-1,j*(padding+W.shape[4]):(j+1)*(padding+W.shape[4])-1] = W[i,j,:,:,:]
    return visualize_matrix(W_full,ubound=ubound)

def visualize_matrix(X,interval_len=50,ubound = 1):
    fig = plt.figure()
    ax = plt.gca()
    im = plt.imshow(X[0,:,:], cmap='Greys_r', vmin=-ubound, vmax=ubound,interpolation='none')
    def up(t):
        im.set_array(X[t,:,:])
        return im,
    ani = anim.FuncAnimation(fig, up, xrange(X.shape[0]), interval=interval_len, blit=True, repeat_delay=1000)
    plt.show()
    return ani

def visualize_matrix2(X,interval_len=50,mymin,mymax):
    fig = plt.figure()
    ax = plt.gca()
    im = plt.imshow(X[0,:,:], cmap='Greys_r', vmin=mymin, vmax=mymax,interpolation='none')
    def up(t):
        im.set_array(X[t,:,:])
        return im,
    ani = anim.FuncAnimation(fig, up, xrange(X.shape[0]), interval=interval_len, blit=True, repeat_delay=1000)
    plt.show()
    return ani

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def feature_inversion(model_weights,layer_names):
    out = []
    # Input a model, copies the model structure (for now just has a fixed structure)
    # copies weights into the model
    # builds feature inversion functions for each layer_name in layer_names []
    # returns feature inverted images in an []
    
    # step 1: build model
    
    # working on gradient from: http://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

    nb_c = [3,3,3]
    nb_p = [2,2]
    nb_f = [4,4,4]
    input_img_shape = (1,1, 16, 32, 32)
    input_img = K.placeholder(input_img_shape)

    model = Sequential()
    first_layer = ZeroPadding3D((0,1,1),input_shape=(1,16,32,32), dim_ordering='th')
    first_layer.input = input_img
    model.add(first_layer)
    model.add(Convolution3D(nb_f[0],len_conv_dim1=1, len_conv_dim2=nb_c[0], len_conv_dim3=nb_c[0], border_mode='valid',
                             activation='relu', name='LGN'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, nb_p[0], nb_p[0])))
    #model.add(Dropout(0.5))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Convolution3D(nb_f[1],len_conv_dim1=nb_c[1], len_conv_dim2=nb_c[1], len_conv_dim3=nb_c[1], border_mode='valid',
                            activation='relu', name='V1'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(nb_p[1], nb_p[1], nb_p[1])))
    #model.add(Dropout(0.5))
    #model.add(ZeroPadding3D((1,1,1)))
    model.add(Convolution3D(nb_f[2],len_conv_dim1=nb_c[2], len_conv_dim2=nb_c[2], len_conv_dim3=nb_c[2], border_mode='valid',
                            activation='relu', name='MT'))

    # IGNORE LAST LAYERS
    #model.add(BatchNormalization())
    #model.add(Flatten())
    ##model.add(Dropout(0.5))
    ##model.add(Dense(4, init='normal', activation='relu', W_regularizer=l2(reg)))
    #model.add(Dense(nb_classes, init='normal', W_regularizer=l2(r)))
    #model.add(Activation('softmax'))

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    for li in range(len(model.layers)):
        model.layers[li].set_weights(model_weights.layers[li].get_weights())

    for layer_name in layer_names:
        print 'Running layer: ', layer_name
        f_images = np.zeros((4,16,32,32))
        for filter_index in range(4):
            # build a loss function that maximizes the activation
            # of the nth filter of the layer considered
            layer_output = layer_dict[layer_name].get_output()
            loss = K.mean(layer_output[:, filter_index, :, :,:])

            # compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img], [loss, grads])
            
            step = 1000
            # we start from a gray image with some noise
            input_img_data = np.random.randn(1,1,16,32,32) * 20 + 128.
            # run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step
            
            img = input_img_data[0,0,:,:,:]
            img = deprocess_image(img)
            
            f_images[filter_index,:,:,:] = img
            
        out.append(f_images)
        
    return out

def obtain_output(model_weights,layer_name,X_train,X_val,X_test):
    out = []
    # Input a model, copies the model structure (for now just has a fixed structure)
    # copies weights into the model
    # builds feature inversion functions for each layer_name in layer_names []
    # returns feature inverted images in an []
    
    # step 1: build model
    
    # working on gradient from: http://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

    nb_c = [3,3,3]
    nb_p = [2,2]
    nb_f = [4,4,4]
    input_img_shape = (1,1, 16, 32, 32)
    input_img = K.placeholder(input_img_shape)

    model = Sequential()
    first_layer = ZeroPadding3D((0,1,1),input_shape=(1,16,32,32), dim_ordering='th')
    first_layer.input = input_img
    model.add(first_layer)
    model.add(Convolution3D(nb_f[0],len_conv_dim1=1, len_conv_dim2=nb_c[0], len_conv_dim3=nb_c[0], border_mode='valid', activation='relu', name='LGN'))
    if(layer_name == 'V1' or layer_name == 'MT'):
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(1, nb_p[0], nb_p[0])))
        #model.add(Dropout(0.5))
        model.add(ZeroPadding3D((1,1,1)))
        model.add(Convolution3D(nb_f[1],len_conv_dim1=nb_c[1], len_conv_dim2=nb_c[1], len_conv_dim3=nb_c[1], border_mode='valid', activation='relu', name='V1'))
        if(layer_name == 'MT'):
            model.add(BatchNormalization())
            model.add(MaxPooling3D(pool_size=(nb_p[1], nb_p[1], nb_p[1])))
            #model.add(Dropout(0.5))
            #model.add(ZeroPadding3D((1,1,1)))
            model.add(Convolution3D(nb_f[2],len_conv_dim1=nb_c[2], len_conv_dim2=nb_c[2], len_conv_dim3=nb_c[2], border_mode='valid', activation='relu', name='MT'))
    
    l = 1e-3#10**(-1*(np.random.rand()*2+4))
    sgd = RMSprop(lr=l, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')

    # IGNORE LAST LAYERS
    #model.add(BatchNormalization())
    #model.add(Flatten())
    ##model.add(Dropout(0.5))
    ##model.add(Dense(4, init='normal', activation='relu', W_regularizer=l2(reg)))
    #model.add(Dense(nb_classes, init='normal', W_regularizer=l2(r)))
    #model.add(Activation('softmax'))

    for li in range(len(model.layers)):
        model.layers[li].set_weights(model_weights.layers[li].get_weights())

    print 'Running layer: ', layer_name
    
    Z_train = model.predict_proba(X_train)
    Z_val = model.predict_proba(X_val)
    Z_test = model.predict_proba(X_test)
    return Z_train, Z_val, Z_test
