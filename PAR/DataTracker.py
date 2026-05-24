import json
import os
import torch
import csv
import json

class DataTracker:
    def __init__(self, 
                 training_metrics_file="./models/training_metrics.csv",
                 validation_metrics_file="./models/validation_metrics.csv", 
                 training_losses_file='./models/training_loss.json',
                 validation_losses_file='./models/validation_loss.json',
                 tn_fp_fn_tp_file_train="./models/tn_fp_fn_tp_train.csv",
                 tn_fp_fn_tp_file_val="./models/tn_fp_fn_tp_val.csv",
                 verbose=False):
        
        self.verbose = verbose
        self.counter = {
            'training': {
                'gender': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
                'bag': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
                'hat': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            },
            'validation': {
                'gender': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
                'bag': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
                'hat': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            }
        }
        self.metrics = {
            'training': {
                'recall': {'gender': [], 'bag': [], 'hat': []},
                'precision': {'gender': [], 'bag': [], 'hat': []},
                'F1': {'gender': [], 'bag': [], 'hat': []},
                'accuracy': {'gender': [], 'bag': [], 'hat': []},
                'balanced_accuracy': {'gender': [], 'bag': [], 'hat': []}
            },
            'validation': {
                'recall': {'gender': [], 'bag': [], 'hat': []},
                'precision': {'gender': [], 'bag': [], 'hat': []},
                'F1': {'gender': [], 'bag': [], 'hat': []},
                'accuracy': {'gender': [], 'bag': [], 'hat': []},
                'balanced_accuracy': {'gender': [], 'bag': [], 'hat': []}
            }
        }
        self.training_loss = {
            'multitask': [],
            'gender': [],
            'hat': [],
            'bag': []
        }
        self.validation_loss = {
            'multitask': [],
            'gender': [],
            'hat': [],
            'bag': []
        }

        self.eps = 1e-8
        self.metrics_files = {
            'training': training_metrics_file,
            'validation': validation_metrics_file
        }
        self.losses_files = {
            'training': training_losses_file,
            'validation': validation_losses_file
        }
        self.tn_fp_fn_tp_file = {
            'training' : tn_fp_fn_tp_file_train,
            'validation' : tn_fp_fn_tp_file_val
        }

        # Initialize metrics files if they don't exist
        for phase in ['training', 'validation']:
            if self.metrics_files[phase] is not None:
                if not os.path.isfile(self.metrics_files[phase]):
                    with open(self.metrics_files[phase], mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            f"Gender_Accuracy", f"Gender_Precision", f"Gender_Recall", f"Gender_F1", f"Gender_Balanced_Accuracy",
                            f"Bag_Accuracy", f"Bag_Precision", f"Bag_Recall", f"Bag_F1", f"Bag_Balanced_Accuracy",
                            f"Hat_Accuracy", f"Hat_Precision", f"Hat_Recall", f"Hat_F1", f"Hat_Balanced_Accuracy"
                        ])
            if self.tn_fp_fn_tp_file[phase] is not None:
                if not os.path.isfile(self.tn_fp_fn_tp_file[phase]):
                    with open(self.tn_fp_fn_tp_file[phase], mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Task', 'TN', 'FP', 'FN', 'TP'])
        

        # Load existing loss files if they exist
        self.load_loss_files()

    def _balanced_accuracy(self, phase='training'):
        """
        Calculates the balanced accuracy: (recall + specificity) / 2.
        Specificity = TN / (TN + FP)

        Parameters:
        phase (str): The phase for which the balanced accuracy is calculated. 
                      Default is 'training'.

        Returns:
        None. The calculated balanced accuracy is stored in the 'balanced_accuracy' dictionary.
        """
        for task in self.counter[phase].keys():
            recall = self.counter[phase][task]['TP'] / (self.counter[phase][task]['TP'] + self.counter[phase][task]['FN'] + self.eps)
            specificity = self.counter[phase][task]['TN'] / (self.counter[phase][task]['TN'] + self.counter[phase][task]['FP'] + self.eps)
            self.metrics[phase]['balanced_accuracy'][task].append((recall + specificity) / 2)

    def _recall(self, phase='training'):
        """
        Calculates the recall for each task in the specified phase.

        Parameters:
        phase (str): The phase for which the recall is calculated. 
                      Default is 'training'.

        Returns:
        None. The calculated recall is stored in the 'recall' dictionary.
        """
        for task in self.counter[phase].keys():
            self.metrics[phase]['recall'][task].append(
                self.counter[phase][task]['TP'] / (self.counter[phase][task]['TP'] + self.counter[phase][task]['FN'] + self.eps)
            )


    def _accuracy(self, phase='training'):
        '''
        Calculates the accuracy for each task in the specified phase.
        Accuracy is defined as (TP+TN) / (TP+TN+FP+FN).

        Parameters:
        phase (str): The phase for which the accuracy is calculated. 
                      Default is 'training'.

        Returns:
        None. The calculated accuracy is stored in the 'accuracy' dictionary.
        '''
        for task in self.counter[phase].keys():
            self.metrics[phase]['accuracy'][task].append(
                (self.counter[phase][task]['TP'] + self.counter[phase][task]['TN']) / 
                (self.counter[phase][task]['TP'] + self.counter[phase][task]['TN'] + 
                 self.counter[phase][task]['FN'] + self.counter[phase][task]['FP'] + self.eps)
            )


    def _precision(self, phase='training'):
        '''
        Calculates the precision for each task in the specified phase.
        Precision is defined as TP / (TP + FP).

        Parameters:
        phase (str): The phase for which the precision is calculated. 
                      Default is 'training'.

        Returns:
        None. The calculated precision is stored in the 'precision' dictionary.
        '''
        for task in self.counter[phase].keys():
            self.metrics[phase]['precision'][task].append(
                self.counter[phase][task]['TP'] / (self.counter[phase][task]['TP'] + self.counter[phase][task]['FP'] + self.eps)
            )


    def _F1score(self, phase='training'):
        '''
        Calculates the F1 score for each task in the specified phase.
        The F1 score is defined as (2*precision*recall)/(precision+recall).

        Parameters:
        phase (str): The phase for which the F1 score is calculated. 
                      Default is 'training'.

        Returns:
        None. The calculated F1 score is stored in the 'F1' dictionary.
        '''
        for task in self.counter[phase].keys():
            recall = self.metrics[phase]['recall'][task][-1]
            precision = self.metrics[phase]['precision'][task][-1]
            self.metrics[phase]['F1'][task].append((2*recall*precision)/(precision+recall+self.eps))



    def save_tn_fp_fn_tp(self, phase="training"):
        """
        Save True Negative (TN), False Positive (FP), False Negative (FN), and True Positive (TP) metrics to a CSV file.

        Parameters:
        phase (str): The phase for which the metrics are being saved. Default is 'training'.

        Returns:
        None. The metrics are saved to a CSV file specified by `self.tn_fp_fn_tp_file[phase]`.
        """
        with open(self.tn_fp_fn_tp_file[phase], mode='a', newline='') as file:
            writer = csv.writer(file)
            for task, counts in self.counter[phase].items():
                writer.writerow([task, counts['TN'], counts['FP'], counts['FN'], counts['TP']])


    def compute_metrics(self, phase='training'):
        """
        Computes and stores the metrics for the specified phase.

        Parameters:
        phase (str): The phase for which the metrics are calculated. 
                      Default is 'training'.

        Returns:
        None. The calculated metrics are stored in the 'metrics' dictionary.
        """
        if self.verbose:
            print(f"\nBefore computing {phase} metrics:")
            for task, counts in self.counter[phase].items():
                print(f"{task.capitalize()} counts: TP={counts['TP']}, TN={counts['TN']}, FP={counts['FP']}, FN={counts['FN']}")

        self._recall(phase)
        self._accuracy(phase)
        self._precision(phase)
        self._F1score(phase)
        self._balanced_accuracy(phase)

        self.save_tn_fp_fn_tp(phase)
        self._clear_count(phase)
        self._append_last_to_file(phase)

    
    def update_counter(self, logits, target, phase='training', threshold=None):
        """
        Updates the counter for the specified phase based on the predictions made by the model.

        Parameters:
        logits (list of torch.Tensor): The model's predictions for each task.
        target (torch.Tensor): The ground truth labels for each task.
        phase (str): The phase for which the counter is being updated. Default is 'training'.
        threshold (list of float): The threshold values for each task. If not provided, the predictions are based on the argmax of the softmax probabilities.

        Returns:
        None. The counter is updated in-place.
        """
        gender_preds = None
        bag_preds = None
        hat_preds = None

        if threshold is not None:
            gender_preds = (torch.nn.functional.softmax(logits[0], dim=1)[:, 1] >= threshold[0])
            bag_preds = (torch.nn.functional.softmax(logits[1], dim=1)[:, 1] >= threshold[1])
            hat_preds = (torch.nn.functional.softmax(logits[2], dim=1)[:, 1] >= threshold[2])
        else:
            logits = [torch.nn.functional.softmax(logit, dim=1) for logit in logits]

            gender_preds = logits[0].argmax(dim=1)
            bag_preds = logits[1].argmax(dim=1)
            hat_preds = logits[2].argmax(dim=1)

        self.counter[phase]['gender']['TP'] += ((gender_preds == 1) & (target[0] == 1)).sum().item()
        self.counter[phase]['gender']['TN'] += ((gender_preds == 0) & (target[0] == 0)).sum().item()
        self.counter[phase]['gender']['FP'] += ((gender_preds == 1) & (target[0] == 0)).sum().item()
        self.counter[phase]['gender']['FN'] += ((gender_preds == 0) & (target[0] == 1)).sum().item()

        self.counter[phase]['bag']['TP'] += ((bag_preds == 1) & (target[1] == 1)).sum().item()
        self.counter[phase]['bag']['TN'] += ((bag_preds == 0) & (target[1] == 0)).sum().item()
        self.counter[phase]['bag']['FP'] += ((bag_preds == 1) & (target[1] == 0)).sum().item()
        self.counter[phase]['bag']['FN'] += ((bag_preds == 0) & (target[1] == 1)).sum().item()

        self.counter[phase]['hat']['TP'] += ((hat_preds == 1) & (target[2] == 1)).sum().item()
        self.counter[phase]['hat']['TN'] += ((hat_preds == 0) & (target[2] == 0)).sum().item()
        self.counter[phase]['hat']['FP'] += ((hat_preds == 1) & (target[2] == 0)).sum().item()
        self.counter[phase]['hat']['FN'] += ((hat_preds == 0) & (target[2] == 1)).sum().item()


    def _clear_count(self, phase='training'):
        """
        Clears the count for the specified phase.

        Parameters:
        phase (str): The phase for which the count is being cleared. 
                      Default is 'training'.

        Returns:
        None. The count is cleared in-place.
        """
        self.counter[phase] = {
            'gender': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
            'bag': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
            'hat': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        }


    def get_metrics(self, phase='training'):
        """
        Retrieves the metrics for each task in the specified phase.

        Parameters:
        phase (str): The phase for which the metrics are calculated. 
                      Default is 'training'.

        Returns:
        accuracy (dict): The accuracy for each task.
        precision (dict): The precision for each task.
        recall (dict): The recall for each task.
        F1score (dict): The F1 score for each task.
        balanced_accuracy (dict): The balanced accuracy for each task.
        """
        recall = {}
        precision = {}
        F1score = {}
        accuracy = {}
        balanced_accuracy = {}

        for task in self.counter[phase].keys():
            recall[task] = self.metrics[phase]['recall'][task][-1]
            precision[task] = self.metrics[phase]['precision'][task][-1]
            F1score[task] = self.metrics[phase]['F1'][task][-1]
            accuracy[task] = self.metrics[phase]['accuracy'][task][-1]
            balanced_accuracy[task] = self.metrics[phase]['balanced_accuracy'][task][-1]

        return accuracy, precision, recall, F1score, balanced_accuracy


    def _append_last_to_file(self, phase):
        """
        Appends the last calculated metrics for each task in the specified phase to a CSV file.

        Parameters:
        phase (str): The phase for which the metrics are being appended. 
                      Default is 'training'.

        Returns:
        None. The metrics are appended to a CSV file specified by `self.metrics_files[phase]`.
        """
        rows = []
        phase_values = []

        for task in ['gender', 'bag', 'hat']:
            task_metrics = [
                self.metrics[phase]['accuracy'][task][-1] if self.metrics[phase]['accuracy'][task] else None,
                self.metrics[phase]['precision'][task][-1] if self.metrics[phase]['precision'][task] else None,
                self.metrics[phase]['recall'][task][-1] if self.metrics[phase]['recall'][task] else None,
                self.metrics[phase]['F1'][task][-1] if self.metrics[phase]['F1'][task] else None,   
                self.metrics[phase]['balanced_accuracy'][task][-1] if self.metrics[phase]['balanced_accuracy'][task] else None

            ]
            phase_values.extend(task_metrics)

        rows.append(phase_values)

        with open(self.metrics_files[phase], mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)


    def update_training_loss(self, multitask, gender, bag, hat):
        """
        Updates the training loss for each task.

        Parameters:
        multitask (float): The loss value for the multitask task.
        gender (float): The loss value for the gender task.
        bag (float): The loss value for the bag task.
        hat (float): The loss value for the hat task.

        Returns:
        None. The loss values are appended to the respective lists in the `training_loss` dictionary.
        """
        self.training_loss['gender'].append(gender) if gender is not None else self.training_loss['gender'].append(-1)
        self.training_loss['bag'].append(bag) if bag is not None else self.training_loss['bag'].append(-1)
        self.training_loss['hat'].append(hat) if hat is not None else self.training_loss['hat'].append(-1)
        self.training_loss['multitask'].append(multitask) if multitask is not None else self.training_loss['multitask'].append(-1)
        self.write_losses_to_file('training')


    def update_validation_loss(self, multitask, gender, bag, hat):
        """
        Updates the validation loss for each task.

        Parameters:
        multitask (float): The loss value for the multitask task.
        gender (float): The loss value for the gender task.
        bag (float): The loss value for the bag task.
        hat (float): The loss value for the hat task.

        Returns:
        None. The loss values are appended to the respective lists in the `validation_loss` dictionary.
        """
        self.validation_loss['gender'].append(gender) if gender is not None else self.validation_loss['gender'].append(-1)
        self.validation_loss['bag'].append(bag) if bag is not None else self.validation_loss['bag'].append(-1)
        self.validation_loss['hat'].append(hat) if hat is not None else self.validation_loss['hat'].append(-1)
        self.validation_loss['multitask'].append(multitask) if multitask is not None else self.validation_loss['multitask'].append(-1)
        self.write_losses_to_file('validation')


    def get_training_losses(self):
        return self.training_loss

    def get_validation_losses(self):
        return self.validation_loss

    def write_losses_to_file(self, phase):
        data = self.training_loss if phase == 'training' else self.validation_loss
        with open(self.losses_files[phase], mode='w') as file:
            json.dump(data, file, indent=4)

    def load_loss_files(self):
        # Load training losses
        if self.losses_files['training'] is not None:
            if os.path.exists(self.losses_files['training']):
                with open(self.losses_files['training'], mode='r') as file:
                    self.training_loss = json.load(file)
        
        # Load validation losses
        if self.losses_files['validation'] is not None:
            if os.path.exists(self.losses_files['validation']):
                with open(self.losses_files['validation'], mode='r') as file:
                    self.validation_loss = json.load(file)