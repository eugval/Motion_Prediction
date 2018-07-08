
class EarlyStopper(object):

    def __init__(self, patience, after_stop_training = 0, verbose = 1):
        '''
        Early Stopper class, checkpoints the model at each epoch and triggers stop training if val set performance
        has been decreasing over a number of epochs
        :param patience: number of epochs where performance needs to be decreasing to trigger early stopping.
        :param after_stop_training: number of epochskeep training for a number of epochs after stopping,
         just for illustration purposes without saving the model.
        :param verbose: verbosity of print mesages
        '''

        self.patience = patience
        self.decreasing_epochs = 0
        self.last_increasing_val = 0.0
        self.verbose = verbose

        self.stop = False
        self.after_stop_training = after_stop_training

    def checkpoint(self, performance_metric):
        '''
        Checks whether to trigger early stopping and whether to save the model.
        :param performance_metric: the metric used to monitor early stopping
        :return: True if the model is to be saved, false otherwise.
        '''

        #If early stopping is not triggered and the performance is increasing, save the model
        if (not self.stop and performance_metric >= self.last_increasing_val):
            if (self.verbose>0):
                print("Performance increasing in the val set, saving the model!")
            self.last_increasing_val = performance_metric
            self.decreasing_epochs = 0
            return True
        else:
            #record decreasing performance
            self.decreasing_epochs += 1

            #If performance has been decreasing more than the patience trigger early stopping
            if (self.decreasing_epochs >= self.patience):
                print("Early stopping activated : {} consecutive epochs where performance is decreasing!".format(
                    self.decreasing_epochs))
                print("Model saved yields a val set accuracy of : {}".format(self.last_increasing_val))
                self.stop = True
                return False
            else:
                #Else just continue training but don't save the model
                if (self.verbose>0):
                    print("Performance is decreasing on the val set, taking note and continuing!")
                return False

    def continue_training(self):
        '''
        Checks whether training is to be contunued or not
        :return: False if breaking out of training loop, true otherwise
        '''
        if(not self.stop):
            return True
        elif(self.stop and self.after_stop_training <=0):
            return False
        else:
            self.after_stop_training -= 1
            return True
