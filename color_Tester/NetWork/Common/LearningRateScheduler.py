import tensorflow as tf


class StepLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, base_lr, end_lr, lrd, lr_decay):
        super(StepLearningRateScheduler, self).__init__()
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.lrd = lrd
        self.lr_decay = lr_decay
        self.cur_lr = 0
        self.cur_step = 0

    def on_epoch_end(self, batch, logs=None):
        if self.cur_step % self.lrd == 0 and self.cur_step != 0:
            self.cur_lr = self.cur_lr * self.lr_decay
            self.cur_lr = max(self.cur_lr, self.end_lr)
        elif self.cur_step == 0:
            self.cur_lr = self.base_lr

        tf.keras.backend.set_value(self.model.optimizer.lr, self.cur_lr)
        self.cur_step += 1
        print("Step: " + str(self.cur_step) + ", Cur LR: " + str(self.cur_lr))

    def reset(self):
        self.cur_step = 0
