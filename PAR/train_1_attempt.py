import os

import torch.optim as optim
import torch.profiler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchvision.transforms as T
from loss_fn.MaskedLoss import *
from net.MultitaskNet import *
import argparse
from tqdm import tqdm
from Dataset import *
from DataTracker import DataTracker
from CustomSampler import *
from GradNorm import GradNorm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
static_task_weight = [1.0, 1.0, 1.0]

def _train(model, trainLoader, optimizer, metrics, class_weight, gradNorm, initial_loss):
    loss_fn = [
        CrossEntropyLoss(weight=class_weight[0].to(device), reduction='mean'),
        CrossEntropyLoss(weight=class_weight[1].to(device), reduction='mean'),
        CrossEntropyLoss(weight=class_weight[2].to(device), reduction='mean')
    ]

    maskedLoss  = MaskedLoss()

    gender_ep_loss = 0
    bag_ep_loss    = 0
    hat_ep_loss    = 0
    total_ep_loss  = 0

    model.train()

    with tqdm(total=len(trainLoader), position=0, leave=True) as epoch_pbar:
        batch_count = 0
        for inputs, gender_labels, bag_labels, hat_labels in trainLoader:
            inputs, gender_labels, bag_labels, hat_labels = (
                inputs.to(device),
                gender_labels.to(device),
                bag_labels.to(device),
                hat_labels.to(device)
            )    
            batch_count += 1

            # Combined forward pass to compute all losses
            gender_logit, bag_logit, hat_logit = model(inputs)
            
            # Compute all losses
            gender_loss = maskedLoss(gender_logit, gender_labels, loss_fn[0])
            bag_loss = maskedLoss(bag_logit, bag_labels, loss_fn[1])
            hat_loss = maskedLoss(hat_logit, hat_labels, loss_fn[2])
            
            task_losses  = torch.stack([gender_loss, bag_loss, hat_loss])
            if initial_loss is None:
                initial_loss = task_losses.clone().detach()

            weighted_loss = gradNorm.compute_weighted_loss(task_losses)

            optimizer.zero_grad()
            weighted_loss.backward(retain_graph=True)
            gradNorm.gradnorm_step(task_losses, initial_loss)
            optimizer.step()
            
            
            # Update metrics
            with torch.no_grad():
                metrics.update_counter(
                    [gender_logit, bag_logit, hat_logit],
                    [gender_labels, bag_labels, hat_labels],
                    phase='training'
                )
                                
                gender_ep_loss += task_losses[0].item()
                hat_ep_loss += task_losses[1].item()
                bag_ep_loss += task_losses[2].item()
                total_ep_loss += weighted_loss.item()

            epoch_pbar.set_postfix(loss=weighted_loss.item())
            epoch_pbar.update(1)
        
        gender_ep_loss /= batch_count
        hat_ep_loss    /= batch_count
        bag_ep_loss    /= batch_count
        total_ep_loss  /= batch_count

        metrics.compute_metrics(phase="training")
        metrics.update_training_loss(total_ep_loss, gender_ep_loss, bag_ep_loss, hat_ep_loss)
        torch.cuda.empty_cache()

        return total_ep_loss, gender_ep_loss, bag_ep_loss, hat_ep_loss
def _validation(model, valLoader, metrics, class_weight, loss_weight):
    with torch.no_grad():
        loss_fn = [
            CrossEntropyLoss(weight=class_weight[0].to(device), reduction='mean'),
            CrossEntropyLoss(weight=class_weight[1].to(device), reduction='mean'),
            CrossEntropyLoss(weight=class_weight[2].to(device), reduction='mean')
        ]

        maskedLoss  = MaskedLoss()


        gender_ep_loss = 0
        bag_ep_loss    = 0
        hat_ep_loss    = 0
        total_ep_loss  = 0

        model.eval()

        with tqdm(total=len(valLoader), position=0, leave=True) as epoch_pbar:
            batch_count=0
            for inputs, gender_labels, bag_labels, hat_labels in valLoader:
                inputs, gender_labels, bag_labels, hat_labels = (
                    inputs.to(device),
                    gender_labels.to(device),
                    bag_labels.to(device),
                    hat_labels.to(device)
                )
                batch_count += 1

                gender_logit, bag_logit, hat_logit = model(inputs)
                metrics.update_counter([gender_logit, bag_logit, hat_logit], [gender_labels, bag_labels, hat_labels], phase="validation")

                # Compute the loss functions
                gender_loss = maskedLoss(gender_logit, gender_labels, loss_fn[0])
                bag_loss    = maskedLoss(bag_logit, bag_labels, loss_fn[1])
                hat_loss    = maskedLoss(hat_logit, hat_labels, loss_fn[2])
                weighted_loss = loss_weight[0]*gender_loss + loss_weight[1]*bag_loss + loss_weight[2]*hat_loss

                gender_ep_loss += gender_loss.item()
                hat_ep_loss    += hat_loss.item()
                bag_ep_loss    += bag_loss.item()
                total_ep_loss  += weighted_loss.item()

                epoch_pbar.set_postfix(loss=weighted_loss.item())
                epoch_pbar.update(1)
            
            gender_ep_loss /= batch_count
            hat_ep_loss    /= batch_count
            bag_ep_loss    /= batch_count
            total_ep_loss  /= batch_count
            metrics.compute_metrics(phase='validation')
            metrics.update_validation_loss(total_ep_loss, gender_ep_loss, bag_ep_loss, hat_ep_loss)
            return total_ep_loss, gender_ep_loss, bag_ep_loss, hat_ep_loss


def training_loop(model, optmizer, loaders, epochs_to_train, weights):

    metrics = DataTracker(
        tn_fp_fn_tp_file_train='./a1_models/tn_fp_fn_tp_train.csv',
        tn_fp_fn_tp_file_val='./a1_models/tn_fp_fn_tp_val.csv',
        training_losses_file='./a1_models/training_losses.json',
        validation_losses_file='./a1_models/validation_losses.json',
        validation_metrics_file='./a1_models/validation_metrics.csv',
        training_metrics_file='./a1_models/training_metrics.csv'
    )

    prv_epoch = 0 if len(metrics.get_training_losses()['multitask'])==0 else len(metrics.get_training_losses()['multitask'])+1    
    best_loss = float('inf') if len(metrics.get_training_losses()['multitask'])==0 else min(metrics.get_training_losses()['multitask'])    
    gradNorm = GradNorm(model.get_last_shared_layers(), 3, device)
    initial_loss = None
    print("------- Initial Parameters -----")
    print(f"Static Task Weights: {static_task_weight}")
    print(f"Optimizer : {optmizer}")
    print(f"Epochs to Train : {epochs_to_train}")
    print(f"Already trained epochs : {prv_epoch}")
    print(f"Best Loss : {best_loss}")
    print(f"Gender class weight : {weights[0]}")
    print(f"Bag class weight : {weights[1]}")
    print(f"Hat class weight : {weights[2]}")

    print("\n------- Training Start -----\n")
    for epoch in range(prv_epoch, epochs_to_train):
        print(f"Epoch {epoch}/{epochs_to_train}...")
        # Training step
        train_losses = _train(model, loaders[0], optmizer, metrics, weights, gradNorm, initial_loss)
        print("------- Training Results -----")
        print(f"Task loss weights : {gradNorm.get_weights()}")
        print(f"Loss:")
        print(f"\tMultitask : {train_losses[0]}")
        print(f"\tGender : {train_losses[1]}")
        print(f"\tBag : {train_losses[2]}")
        print(f"\tHat : {train_losses[3]}")
        accuracy, precision, recall, f1Score, balanced_accuracy = metrics.get_metrics(phase='training')
        print("Metrics:")
        print(f"\tGender :\n\t\trecall : {recall['gender'] * 100}\n\t\taccuracy : {accuracy['gender'] * 100}\n\t\tF1Score : {f1Score['gender'] * 100}\n\t\tPrecision : {precision['gender'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['gender']}")
        print(f"\n\tBag : \n\t\trecall : {recall['bag'] * 100}\n\t\taccuracy : {accuracy['bag'] * 100}\n\t\tF1Score : {f1Score['bag'] * 100}\n\t\tPrecision : {precision['bag'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['bag']}")
        print(f"\n\tHat : \n\t\trecall : {recall['hat'] * 100}\n\t\taccuracy : {accuracy['hat'] * 100}\n\t\tF1Score : {f1Score['hat'] * 100}\n\t\tPrecision : {precision['hat'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['hat']}")
    
        # Validation step
        losses = _validation(model, loaders[1], metrics, weights, gradNorm.get_weights())
        print(f"\n------- Validation Results -----")
        print(f"Loss:")
        print(f"\tMultitask : {losses[0]}")
        print(f"\tGender : {losses[1]}")
        print(f"\tBag : {losses[2]}")
        print(f"\tHat : {losses[3]}")
        accuracy, precision, recall, f1Score, balanced_accuracy = metrics.get_metrics(phase='validation')
        print("Metrics:")
        print(f"\tGender :\n\t\trecall : {recall['gender'] * 100}\n\t\taccuracy : {accuracy['gender'] * 100}\n\t\tF1Score : {f1Score['gender'] * 100}\n\t\tPrecision : {precision['gender'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['gender']}")
        print(f"\n\tBag : \n\t\trecall : {recall['bag'] * 100}\n\t\taccuracy : {accuracy['bag'] * 100}\n\t\tF1Score : {f1Score['bag'] * 100}\n\t\tPrecision : {precision['bag'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['bag']}")
        print(f"\n\tHat : \n\t\trecall : {recall['hat'] * 100}\n\t\taccuracy : {accuracy['hat'] * 100}\n\t\tF1Score : {f1Score['hat'] * 100}\n\t\tPrecision : {precision['hat'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['hat']}")
        
        print("------- Checkpoint Saving -----")
        if train_losses[0] < best_loss: # Compare the best_loss with the multitask loss of the ended epoch
            best_loss = train_losses[0]
            print(f"Improved multitask loss to {best_loss:.4f}, saving the model...")
            model.save_model("./a1_models/model.pth")

        # Saving the validation loss at each epochs
        model.save_model(f"./a1_models/validation/_{epoch}.pth")
        print(f"Saved model at {epoch} for the validation process")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../Dataset/training/image", help="Training images directory")
    parser.add_argument("--labels", type=str, default="../Dataset/training/training_set.txt", help="Training labels directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Batch size")
    parser.add_argument("--load_model", type=str, default="./a1_models/model.pth", help="Number of workers")
    parser.add_argument("--max_samples", "-ms", type=int, default=-1, help="Define the max number of samples to use for the training, default -1 (tutti i campioni)")

    args = parser.parse_args()

    

    # Initialize dataset, dataloader, model, and optimizers
    transform = T.Compose([
        T.Resize((220,96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    
    print("Loading datasets...")
    train_dataset = CustomDataset(args.image, args.labels, transform=transform, max_samples=args.max_samples)
    val_dataset = CustomDataset(args.image.replace("training", "validation"), args.labels.replace("training", "validation"),transform=transform)

    print("Initializing data sampler...")
    train_sampler = CustomSampler(train_dataset, args.batch_size)

    print("Initializing dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=12)


    model = MultiTaskNet()
    loss_dict = {
        'Multitask': [],
        'Gender': [],
        'Hat': [],
        "Bag": []
    }
  

    if os.path.exists(args.load_model):
        model.load_model(args.load_model)
    else:
        os.makedirs("./a1_models", exist_ok=True)
        os.makedirs("./a1_models/validation", exist_ok=True)

    optmizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train the model
    model = model.to(device)
    
    training_loop(model, optmizer, [train_dataloader, val_dataloader], args.epochs, weights=train_dataset.calculate_class_weights())
