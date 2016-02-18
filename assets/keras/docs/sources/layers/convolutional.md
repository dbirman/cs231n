
## Convolution1D

```python
keras.layers.convolutional.Convolution1D(nb_filter, filter_length, 
        init='uniform', activation='linear', weights=None, 
        border_mode='valid', subsample_length=1, 
        W_regularizer=None, b_regularizer=None, W_constraint=None, 
        b_constraint=None, input_dim=None, input_length=None)
```

Convolution operator for filtering neighborhoods of one-dimensional inputs. When using this layer as the first layer in a model, either provide the keyword argument `input_dim` (int, e.g. 128 for sequences of 128-dimensional vectors), or `input_shape` (tuple of integers, e.g. (10, 128) for sequences of 10 vectors of 128-dimensional vectors).

- __Input shape__: 3D tensor with shape: `(nb_samples, steps, input_dim)`.

- __Output shape__: 3D tensor with shape: `(nb_samples, steps, nb_filter)`. `steps` value might have changed due to padding.

- __Arguments__:
    - __nb_filter__: Number of convolution kernels to use (dimensionality of the output).
    - __filter_length__: The extension (spatial or temporal) of each filter.
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights.
    - __border_mode__: 'valid' or 'full'. see scipy.signal.convolve2d.
    - __subsample_length__: factor by which to subsample output.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.
    - __input_dim__: Number of channels/dimensions in the input. Either this argument or the keyword argument `input_shape` must be provided when using this layer as the first layer in a model.
    - __input_length__: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten` then `Dense` layers upstream (without it, the shape of the dense outputs cannot be computed).

---

## Convolution2D

```python
keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, 
        init='glorot_uniform', activation='linear', weights=None, 
        border_mode='valid', subsample=(1, 1),
        W_regularizer=None, b_regularizer=None, W_constraint=None)
```

Convolution operator for filtering windows of two-dimensional inputs. When using this layer as the first layer in a model, provide the keyword argument `input_shape` (tuple of integers, does not include the sample axis), e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

- __Input shape__: 4D tensor with shape: `(nb_samples, channels, rows, cols)`.

- __Output shape__: 4D tensor with shape: `(nb_samples, nb_filter, rows, cols)`. `rows`, `cols` might have changed due to padding.


- __Arguments__:

    - __nb_filter__: Number of convolution filters to use.
    - __nb_row__: Number of rows in the convolution kernel.
    - __nb_col__: Number of columns in the convolution kernel.
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights.
    - __border_mode__: 'valid', 'full', or 'same'. [See scipy.signal.convolve2d](http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html).
    - __subsample__: tuple of length 2. Factor by which to subsample output. Also called strides elsewhere.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.


---

## MaxPooling1D

```python
keras.layers.convolutional.MaxPooling1D(pool_length=2, stride=None, ignore_border=True)
```

- __Input shape__: 3D tensor with shape: `(nb_samples, steps, dim)`.

- __Output shape__: 3D tensor with shape: `(nb_samples, downsampled_steps, dim)`.

- __Arguments__:

    - __pool_length__: factor by which to downscale. 2 will halve the input.
    - __stride__: integer or None. Stride value.
    - __ignore_border__: boolean.

---

## MaxPooling2D

```python
keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), ignore_border=True)
```

- __Input shape__: 4D tensor with shape: `(nb_samples, stack_size, nb_row, nb_col)`.

- __Output shape__: 4D tensor with shape: `(nb_samples, stack_size, new_nb_row, new_nb_col)`.

- __Arguments__:

    - __pool_size__: factor by which to downscale (vertical ds, horizontal ds). (2, 2) will halve the image in each dimension.
    - __ignore_border__: boolean. When True, (5, 5) input with pool_size=(2, 2) will generate a (2, 2) output, (3, 3) otherwise.

