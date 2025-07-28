import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jax import vmap
import jax

def init(hidden_size, random_seed = 0, input_size = 256, output_size = 10, std = 1e-4):

    params = {}        
    # params should contain self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
    # TODO ：initialize the parmas
    #
    
    key = random.PRNGKey(random_seed)
    
    # input weight, bias
    params['W1'], key = normal(key,(input_size, hidden_size), std)
    params['b1'] = jnp.zeros(hidden_size)
    
    # hidden weight, bias
    params['W2'], key = normal(key,(hidden_size, output_size), std)
    params['b2'] = jnp.zeros(output_size)

    #Return: generated parameters
    return params

def normal(key, shape, std):
    key, subkey = random.split(key)
    return random.normal(subkey, shape) * std, key

def forward_pass(params, X, y = None, wd_decay = 0.0):
    
    loss = None
    predict = None
    
    states = {}

    #
    # TODO ： finish hidden layer and class layer
    #
    #
    
    #relu
    def relu(x):
        return jnp.where(x>0,x,0)
    
    #softmax
    def softmax(x):
        expx = jnp.exp(x)
        sum_expx = expx.sum(axis = -1)
        return expx/sum_expx.reshape(-1,1)
    
    # hidden comp with ReLu
    states['h1'] = jnp.dot(X,params['W1']) + params['b1']    
    states['a1'] = relu(states['h1']) 
    
    # output comp with softmax
    states['h2'] = jnp.dot(states['a1'],params['W2']) + params['b2'] 
    states['a2'] = softmax(states['h2']) 
    
    def cross_entropy_error(x, y):
        num = states['a2'].shape[1]
        y = jnp.eye(num)[y]
        return -jnp.average(jnp.sum(y*jnp.log(jnp.maximum(x, 1e-12)), axis=1))
    loss = cross_entropy_error(states['a2'],y) + wd_decay/2*(jnp.sum(jnp.square(params['W1']))+jnp.sum(jnp.square(params['W2'])))
    
    predict = jnp.argmax(states['a2'],axis=1)
        
        
    # Returns:
    #   loss: value of loss function
    #   predict: An array of predicted labels of input X
    #   states: A dict, containing intermediate results for computing backprop
    
    return loss, predict, states

def back_prop(params, states, X, y, wd_decay = 0.0):
    grads = {}

    #grads should contain grads['W1'] grads['b1'] grads['W2'] grads['b2']
    
    #
    # TODO
    #
    num = states['a2'].shape[1]
    y = jnp.eye(num)[y]
    derivation_softmax_cross_entropy = states['a2']-y
    
    n = X.shape[0]
    
    
    grads['W2'] = jnp.dot(states['a1'].T, derivation_softmax_cross_entropy)/n + wd_decay*params['W2']
    grads['b2'] = jnp.sum(derivation_softmax_cross_entropy, axis=0, keepdims=True)/n
    
    dl = jnp.dot(derivation_softmax_cross_entropy, params['W2'].T) * jnp.heaviside(states['h1'],0)
    grads['W1'] = jnp.dot(X.T, dl)/n + wd_decay*params['W1']
    grads['b1'] = jnp.sum(dl, axis=0, keepdims=True)/n
    
    return grads

def numerical_gradient(params, X, y, wd_decay = 0.0, delta = 1e-6):
    grads = {}
        
    for param_name in params:

        if jnp.ndim(params[param_name]) == 1:
            idx = jnp.arange(params[param_name].shape[0])
            grads[param_name] = jax.vmap(euler_1d,
                in_axes = (None, None, 0, None, None, None, None),
                out_axes= 0)(params, param_name, idx, X, y, wd_decay, delta)
        else:
            idx = jnp.arange(params[param_name].shape[0])
            idy = jnp.arange(params[param_name].shape[1])
            grads[param_name] = jax.vmap(jax.vmap(euler_2d, in_axes=(None, None, None, 0, None, None, None, None), out_axes = 0),
                in_axes=(None, None, 0, None, None, None, None, None), out_axes = 0)(params, param_name, idx, idy, X, y, wd_decay, delta)

    return grads

def euler_1d(params, param_name, idx, X, y, wd_decay, delta):
    
    old_param = params[param_name][idx]
    params[param_name] = params[param_name].at[idx].set(old_param + delta)
    loss_add_delta = forward_pass(params, X, y, wd_decay)[0]
    params[param_name] = params[param_name].at[idx].set(old_param - delta)
    loss_minus_delta = forward_pass(params, X, y, wd_decay)[0]

    return (loss_add_delta - loss_minus_delta) / (delta * 2)

def euler_2d(params, param_name, idx, idy, X, y, wd_decay, delta):
    old_param = params[param_name][idx, idy]
    params[param_name] = params[param_name].at[idx, idy].set(old_param + delta)
    loss_add_delta = forward_pass(params, X, y, wd_decay)[0]
    params[param_name] = params[param_name].at[idx, idy].set(old_param - delta)
    loss_minus_delta = forward_pass(params, X, y, wd_decay)[0]

    return (loss_add_delta - loss_minus_delta) / (delta * 2)

def train(params, X, y, X_val, y_val, random_seed = 0,
                learning_rate=0, lr_decay = 1,
                momentum=0, do_early_stopping=False,
                wd_decay=0, num_iters=10, alpha = 0,
                batch_size=4, verbose=False, print_every=10, min_delta=0, patience=1):
    # two more params
    # min_delta as the minimum value of progress, if the epoch acc < progress of last epoch, regard as no progress this epoch
    # define patience: if over patience no progress, early stopping
    
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    loss_history = []
    acc_history = []
    val_acc_history = []
    val_loss_history = []
    
    #
    # TODO : training process
    #
    # Hint ：Could initialize some parameters here
    #        Like velocitys or initial parameters
    #
    
    v_W1 = jnp.zeros_like(params['W1'])
    v_W2 = jnp.zeros_like(params['W2'])
    v_b1 = jnp.zeros_like(params['b1'])
    v_b2 = jnp.zeros_like(params['b2'])
    
    no_progress = 0
    
    key = random.PRNGKey(random_seed)

    for it in range(num_iters):

        X_batch = None
        y_batch = None
        
        #
        # TODO : training process
        #
        # Hint:
        # X_batch = ...
        # y_batch = ...
        # Loss = ...
        # grads = ...
        # val_los = ...
        
        key, subkey = random.split(key)
        X_and_y = jnp.append(X,y.reshape(-1,1),axis=1)
        X_and_y_batch = random.choice(subkey,X_and_y,shape=(batch_size,),axis=0)
        X_batch = X_and_y_batch[:,:-1]
        y_batch = X_and_y_batch[:,-1].astype(int)
       
        loss, predict, states = forward_pass(params, X_batch, y_batch, wd_decay = wd_decay)
        grads = back_prop(params, states, X_batch, y_batch, wd_decay = wd_decay)
        
        v_W2 = momentum * v_W2 + grads['W2']
        v_b2 = momentum * v_b2 + grads['b2']
        params['W2'] -= learning_rate * v_W2
        params['b2'] -= learning_rate * v_b2
        
        v_W1 = momentum * v_W1 + grads['W1']
        v_b1 = momentum * v_b1 + grads['b1']
        params['W1'] -= learning_rate * v_W1
        params['b1'] -= learning_rate * v_b1

        if verbose and it % print_every == 0:
            val_loss,_,_ = forward_pass(params,X_val,y_val,wd_decay=wd_decay)
            val_loss_history.append(val_loss) 
            loss_history.append(loss)
            print('iteration %d / %d: training loss %f val loss: %f' % (it, num_iters, loss, val_loss))

        if it % iterations_per_epoch == 0:
            
            train_acc = get_acc(params, X_batch, y_batch)
            val_acc = get_acc(params, X_val, y_val)
            acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            # Decay learning rate
            learning_rate *= lr_decay
            
            if do_early_stopping:
                if len(val_acc_history)>1:
                    progress = val_acc_history[-1] - val_acc_history[-2]
                    if progress < min_delta:
                        no_progress += 1
                    else:
                        no_progress = 0
                    if no_progress > patience:
                        break

    return {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'acc_history': acc_history,
        'val_acc_history': val_acc_history,
    }, params

def get_acc(params, X, y):
    _, predict, _ = forward_pass(params, X, y)
    return jnp.mean(predict == y)