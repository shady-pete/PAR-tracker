import torch
import numpy as np
import torch.optim as optim


class GradNorm:
    '''
        Implementation of the GradNorm algorithm following the paper:
        "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" 
        available at https://arxiv.org/pdf/1711.02257

    '''
    def __init__(self, shared_layer, num_tasks, device, alpha=1.5, lr=0.001, initial_weights=None, verbose=False):
        '''
        Args:
            shared_layers: List containing the shared layers to be used for GradNorm.
            num_tasks: Number of tasks the network performs.
            device: Device on which to create and operate.
            alpha: Hyperparameter required for GradNorm (see paper).
            lr: Learning rate for updating the loss weights.

        '''

        self.layer = shared_layer # Ultimo layer condiviso da utilizzare
        self.num_tasks = num_tasks
        self.alpha = alpha
        if initial_weights is None:
            self.task_weights = torch.nn.Parameter(torch.ones(num_tasks, requires_grad=True, device=device))
            print(f"[GRADNORM] Initial weight is None, Initialize weight from scratch\n\t {self.task_weights}")
        else:
            self.task_weights = torch.nn.Parameter(initial_weights.to(device), requires_grad=True)
            print(f"[GRADNORM] Initial weight is loaded from existings weight \n\t {self.task_weights}")

        self.T = num_tasks # Pari a num task in base al fatto che vengono i pesi vengono inizializzati tutti a 1
        self.optimizer = optim.Adam([self.task_weights], lr=lr)

        self.verbose = verbose

    def get_weights(self):
        return self.task_weights.detach().cpu()

    def compute_task_losses(self, task_outputs, task_targets, loss_fn):
        '''
            Args:
                task_outputs: List containing the outputs obtained from the network.
                task_targets: List containing the Ground Truth for each task.
                loss_fn: Loss function to be used (in this case, it is common for all tasks).

            Returns:
                Tensor containing the losses for each task in a single tensor.

            Example:
                task_losses = [Tensor1, Tensor2, Tensor3]
                torch.stack(task_losses) = Tensor([tensor1.data, tensor2.data, tensor3.data])
        '''
        task_losses = []
        for i in range(self.num_tasks):
            task_losses.append(loss_fn[i](task_outputs[i], task_targets[i] ))
        return torch.stack(task_losses)

    def compute_weighted_loss(self, task_losses):
        '''
            Args:
                task_losses: Column tensor containing the losses for individual tasks.

            Returns:
                Multitask loss obtained as a weighted sum of the weights w_i and the task losses L_i.

            Note: The operator @ represents matrix multiplication (rows x columns) or, in the case of 1D tensors, the dot product.

        '''
        if self.verbose:
            print(f"\n[GRADNORM] Compute weighted loss \n\t weight {self.task_weights}\n\t task_loss {task_losses}\n\t Weighted loss {self.task_weights @ task_losses}")
        return self.task_weights @ task_losses


    def gradnorm_step(self, task_losses, initial_losses):
        '''
        Function responsible for applying the GradNorm algorithm and performing a step 
        in the weight update process.
        
        Args:
            task_losses: Column tensor containing the losses for individual tasks.
            initial_losses: Column tensor containing the initial losses of the tasks.

        '''

        gw = [] # L2 norm calculated on the gradient of the layer of interest with respect to the single Loss.
        for i in range(self.num_tasks):
            params = [param for layer in self.layer for param in layer.parameters()]
            dl = torch.autograd.grad(self.task_weights[i] * task_losses[i], params, retain_graph=True, create_graph=True)[0]
            gw.append(torch.norm(dl))

        gw = torch.stack(gw) # Tensore colonna

        # Tensor
        loss_ratio = task_losses.detach() / initial_losses
        # compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()
        # compute the average gradient norm
        gw_avg = gw.mean().detach()
        # compute the GradNorm loss
        constant = (gw_avg * rt ** self.alpha).detach()
        gradnorm_loss = torch.abs(gw - constant).sum()
        # clear gradients of weights
        self.optimizer.zero_grad()
        # backward pass for GradNorm
        gradnorm_loss.backward(retain_graph=True)
        self.optimizer.step()


        '''
            Following the paper we normalize the weights w(i) so that  
                sum(w(i+1)) = T
        '''
        with torch.no_grad():
            if self.verbose:
                print(f"\n[GRADNORM] Weights before normalization {self.task_weights} - sum to {sum(self.task_weights)}")
            '''
                To modify the value of a Parameters tensor associated with an optimizer, as in this case, I cannot simply assign the new value,
                as this would create a new tensor. Consequently, the optimizer would lose its reference to the original tensor and would no longer
                be able to update the weights. To avoid losing the reference to the tensor, it is necessary to use the copy_() method, which
                allows for an in-place operation to copy the data from a tensor passed as a parameter into the tensor associated with the optimizer.
                Alternatively, the operation can be implemented using operators that perform in-place modifications.
            '''
            self.task_weights.copy_(self.task_weights / self.task_weights.sum() *self.T)
            if self.verbose:
                print(f"\n[GRADNORM] Weights after normalization {self.task_weights} - sum to {sum(self.task_weights)}")

