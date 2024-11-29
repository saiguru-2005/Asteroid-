import numpy as np
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print("--"*20 + f" New Minimunm found {validation_loss} "+"--"*20)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print('*'*20+f' Early Stopping counter : {self.counter}/{self.patience} '+'*'*20)
            print(f'Current change = {validation_loss} from minimum = {self.min_validation_loss}')
            if self.counter >= self.patience:
                
                print(f'Early stopping condition is met............')
                return True
        return False