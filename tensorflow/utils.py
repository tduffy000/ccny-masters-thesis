import os
import time
import tensorflow as tf

# # # # # # #
# CALLBACKS #
# # # # # # #
def lr_scheduler(cutoff, lr):
    def scheduler(epoch):
        if epoch < cutoff:
            return lr
        else:
            return lr * tf.math.exp(0.1 * (cutoff - epoch))
    return scheduler

# we need to add a callback for metadata saving
def get_callback(name, conf, lr=None, freeze=False):
    if name == 'lr_scheduler':
        scheduler = lr_scheduler(conf['cutoff_epoch'], lr)
        return tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)
    elif name == 'csv_logger' and freeze:
        path = f'{conf["dir"]}/{int(time.time())}'
        os.makedirs(path)
        return tf.keras.callbacks.CSVLogger(f'{path}/logs.csv')
    elif name == 'checkpoint' and freeze:
        fpath = f'{conf["dir"]}/{int(time.time())}-weights.hdf5'
        if not os.path.exists(conf['dir']):
            os.makedirs(conf['dir'])
        return tf.keras.callbacks.ModelCheckpoint(fpath)
    return None

# # # # # # #
# OPTIMIZER #
# # # # # # #
def get_optimizer(type, lr, momentum=None, rho=None, epsilon=None, clipnorm=None):
    if type.lower() == 'adam':
        if clipnorm is not None:
            return tf.keras.optimizers.Adam(lr=lr, clipnorm=clipnorm)    
        return tf.keras.optimizers.Adam(lr=lr)
    elif type.lower() == 'sgd':
        if clipnorm is not None:
            return tf.keras.optimizers.SGD(lr=lr, momentum=momentum, clipnorm=clipnorm)    
        return tf.keras.optimizers.SGD(lr=lr, momentum=momentum)
    elif type.lower() == 'rmsprop':
        if clipnorm is not None:
            return tf.keras.optimizers.RMSprop(lr=lr, rho=rho, momentum=momentum, epsilon=epsilon, clipnorm=clipnorm)    
        return tf.keras.optimizers.RMSprop(lr=lr, rho=rho, momentum=momentum, epsilon=epsilon)
    else:
        raise ParameterError()
