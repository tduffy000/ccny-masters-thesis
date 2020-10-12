import os
import time
import tensorflow as tf

# TODO: this can move into the Config object (or perhaps we require a parser?)

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
def get_callback(name, conf, lr=None):
    if name == 'lr_scheduler':
        scheduler = lr_scheduler(conf['cutoff_epoch'], lr)
        return tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)
    elif name == 'csv_logger':
        path = f'{conf["dir"]}/{int(time.time())}'
        os.makedirs(path)
        return tf.keras.callbacks.CSVLogger(f'{path}/logs.csv')
    elif name == 'checkpoint':
        fpath = f'{conf["dir"]}/{int(time.time())}-weights.hdf5'
        if not os.path.exists(conf['dir']):
            os.makedirs(conf['dir'])
        return tf.keras.callbacks.ModelCheckpoint(fpath)
    else:
        raise ParameterError(f'{name} is not a supported callback.')

# # # # # # #
# OPTIMIZER #
# # # # # # #
def get_optimizer(type, lr, momentum=None, rho=None, epsilon=None):
    if type.lower() == 'adam':
        return tf.keras.optimizers.Adam(lr=lr)
    elif type.lower() == 'sgd':
        return tf.keras.optimizers.SGD(lr=lr, momentum=momentum)
    elif type.lower() == 'rmsprop':
        return tf.keras.optimizers.RMSprop(lr=lr, rho=rho, momentum=momentum, epsilon=epsilon)
    else:
        raise ParameterError()
