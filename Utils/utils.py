from datetime import datetime

class TrainingProgress():
    def __init__(self, epochs, step_per_epoch):
        self.epochs = epochs
        self.step_per_epoch = step_per_epoch
    
    def show_progress(self, cur_epoch, cur_step, loss, lr, correct_rate):
        print(f'{datetime.now()} Epoch: {cur_epoch}/{self.epochs}, \
            step: {cur_step}/{self.step_per_epoch} \
            loss: {loss}, learning_rate: {lr}, correct_rate: {correct_rate}')
