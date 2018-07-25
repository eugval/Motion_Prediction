
class SmoothedEarlyStopper(object):

    def __init__(self, patience,  weight_factor = 0.3, seek_decrease = False, after_stop_training = 0, verbose = 1):
        '''
        Early Stopper class, checkpoints the model at each epoch and triggers stop training if val set performance
        has been decreasing over a number of epochs
        :param patience: number of epochs where performance needs to be decreasing to trigger early stopping.
        :param after_stop_training: number of epochskeep training for a number of epochs after stopping,
         just for illustration purposes without saving the model.
        :param verbose: verbosity of print messages
        '''

        self.training_rounds = 0
        self.moving_average = 'start'

        self.weight_factor = weight_factor

        self.stop = False
        self.decreasing_performance = 0
        self.after_stop_training = after_stop_training

        self.patience = patience
        self.seek_decrease = seek_decrease

        self.verbose = verbose


    def checkpoint(self, performance_metric):
        '''
        Checks whether to trigger early stopping and whether to save the model.
        :param performance_metric: the metric used to monitor early stopping
        :return: True if the model is to be saved, false otherwise.
        '''

        if(self.moving_average =='start'):
            self.moving_average = performance_metric

        #If early stopping is not triggered and the performance is increasing, save the model
        if (not self.stop and self.performed_better(performance_metric)):
            if (self.verbose>0):
                print("Performance increasing in the val set, saving the model!")
            self.moving_average = self.weight_factor*performance_metric +(1-self.weight_factor)*self.moving_average
            self.decreasing_performance = 0
            return True
        else:
            #record decreasing performance
            self.decreasing_performance += 1
            self.moving_average = self.weight_factor*performance_metric +(1-self.weight_factor)*self.moving_average

            #If performance has been decreasing more than the patience trigger early stopping
            if (self.decreasing_performance >= self.patience):
                print("Early stopping activated : {} consecutive epochs where performance is decreasing!".format(
                    self.decreasing_performance))
                print("Model saved yields a val set accuracy of : {}".format(self.moving_average))
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

    def performed_better(self, performance_metric):
        if(self.verbose):
            print('moving average performance : {}'.format(self.moving_average))
            print('current performance : {}'.format(performance_metric))

        if(self.seek_decrease):
            return performance_metric <= self.moving_average
        else:
            return performance_metric >= self.moving_average





class EarlyStopper(object):

    def __init__(self, patience, seek_decrease =False, after_stop_training = 0, verbose = 1):
        '''
        Early Stopper class, checkpoints the model at each epoch and triggers stop training if val set performance
        has been decreasing over a number of epochs
        :param patience: number of epochs where performance needs to be decreasing to trigger early stopping.
        :param after_stop_training: number of epochskeep training for a number of epochs after stopping,
         just for illustration purposes without saving the model.
        :param verbose: verbosity of print mesages
        '''

        self.patience = patience
        self.decreasing_performance = 0
        self.last_saved_val = 0.0
        self.verbose = verbose
        self.seek_decrease = seek_decrease

        self.stop = False
        self.after_stop_training = after_stop_training

    def checkpoint(self, performance_metric):
        '''
        Checks whether to trigger early stopping and whether to save the model.
        :param performance_metric: the metric used to monitor early stopping
        :return: True if the model is to be saved, false otherwise.
        '''

        if(self.seek_decrease):
            performance_metric = -performance_metric
        #If early stopping is not triggered and the performance is increasing, save the model
        if (not self.stop and self.performed_better(performance_metric)):
            if (self.verbose>0):
                print("Performance increasing in the val set, saving the model!")
            self.last_saved_val = performance_metric
            self.decreasing_performance = 0
            return True
        else:
            #record decreasing performance
            self.decreasing_performance += 1

            #If performance has been decreasing more than the patience trigger early stopping
            if (self.decreasing_performance >= self.patience):
                print("Early stopping activated : {} consecutive epochs where performance is decreasing!".format(
                    self.decreasing_performance))
                print("Model saved yields a val set accuracy of : {}".format(self.last_saved_val))
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

    def performed_better(self, performance_metric):
        if(self.last_saved_val == 0.0 and performance_metric != 0):
            return True
        else:
            return performance_metric >= self.last_saved_val
