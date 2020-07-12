import tensorflow as tf
import time

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

def get_callback(name, conf, lr=None):
    if name == 'tensorboard':
        return tf.keras.callbacks.TensorBoard(
            log_dir=f'{conf["log_dir"]}/epoch={int(time.time())}',
            write_images=conf['write_images']
        )
    elif name == 'lr_scheduler':
        scheduler = lr_scheduler(conf['cutoff_epoch'], lr)
        return tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)
    elif name == 'csv_logger':
        return tf.keras.callbacks.CSVLogger(f'{conf["path"]}/epoch={int(time.time())}/logs.csv')
    else:
        raise ParameterError('Only callbacks for Tensorboard, CSV logs, and LearningRateScheduler are supported.')

# # # # # # #
# OPTIMIZER #
# # # # # # #
def get_optimizer():
    pass
