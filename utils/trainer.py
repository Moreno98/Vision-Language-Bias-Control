from tqdm import tqdm
import sklearn.metrics as metrics
import torch
import os
import numpy as np

class Trainer(object):
    def __init__(
        self, 
        model,
        optimizer,
        loss,
        save_path,
        device,
        num_attributes,
        opt,
        starting_epoch = 0
    ):
        '''
        Initialize the manager.
        '''
        self.device = device
        self.result_path = save_path
        self.weight_path = os.path.join(self.result_path, "weights")
        self.best_f1_score = 0
        self.starting_epoch = starting_epoch
        self.cls_idx = None
        self.num_attributes = num_attributes
        self.cls_idx = opt['cls_idx']
        self.TARGET_NAMES = [opt['classification']]
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.opt = opt
        
        self.result_file_name = "results.txt"
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.weight_path, exist_ok=True)
        self.log_file = "log.txt"
        
        self.print_info(
            model_name=self.model.__class__.__name__,
            n_params=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            opt=opt
        )

    def train(
        self, 
        train_data_loader, 
        eval_data_loader
    ):
        '''
        Train the model for a given number of epochs.
        Args:
            train_data_loader: The data loader for training.
            eval_data_loader: The data loader for evaluation.
        '''
        epochs = self.opt["epochs"]

        for epoch in range(self.starting_epoch, epochs):
            print("--------------------------------TRAINING--------------------------------")
            print(f"Epoch: {epoch+1}")

            self._update_result_file(
                os.path.join(
                    self.result_path, 
                    self.log_file
                ), 
                [
                    ("--------------------------------TRAINING--------------------------------", ""),
                    ("Epoch: ", epoch+1)
                ]
            )

            # ----------------------------------> train <---------------------------------
            loss_avg_train, train_info = self._run_epoch(
                data_loader=train_data_loader,
                mode="train"
            )

            print("--------------------------------EVALUATION--------------------------------")
            self._update_result_file(
                os.path.join(
                    self.result_path, 
                    self.log_file
                ), 
                [
                    ("--------------------------------EVALUATION--------------------------------", "")
                ]
            )

            # ----------------------------------> eval <---------------------------------
            loss_avg_eval, eval_info = self._run_epoch(
                data_loader=eval_data_loader,
                mode="eval"
            )
                
            if epoch == epochs-1:
                print("Saving last model at epoch", epoch+1)
                self._save_model(
                    self.model,
                    self.optimizer,
                    os.path.join(self.weight_path, f"model_last_epoch_{epoch+1}.pth"),
                    epoch,
                    self.best_f1_score
                )

                # save final results
                self._update_result_file(
                    os.path.join(
                        self.result_path, 
                        self.log_file
                    ), 
                    [
                        ("--------------------------------EVAL FINAL RESULTS--------------------------------", "")
                    ]+eval_info
                )

    def _run_epoch(
        self,
        data_loader,
        mode="eval"
    ):
        '''
        Run an epoch of training or evaluation.
        Args:
            model: The model to use.
            optimizer: The optimizer to use.
            loss: The loss function to use.
            data_loader: The data loader to use.
            mode: The mode to run ("train" or "eval").
        '''
        # use cuda if set
        self.model = self.model.to(self.device)
        # set model to train or eval mode
        # enable gradients if in training mode
        if mode == "train":
            self.model.train()
            self.optimizer.zero_grad()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        progress_bar = tqdm(data_loader, desc="description")
        n_iterations = len(progress_bar)
        # initialize variables
        targets = [[] for i in range(self.num_attributes)]
        preds = [[] for i in range(self.num_attributes)]
        pred_prob = [[] for i in range(self.num_attributes)]
        loss_tasks = [0.0 for i in range(self.num_attributes)]
        protected_labels = [[] for i in range(self.num_attributes)]
        eopp_list = torch.zeros(self.num_attributes, 2, 2)
        eopp_data_count = torch.zeros(self.num_attributes, 2, 2)

        loss_sum = 0
        # run epoch
        for images, labels, protected_targets in progress_bar:
            # augment each image
            images = images.to(self.device)
            # forward pass
            with torch.autocast(device_type=self.device.type):
                outputs = self.model(images)
                # compute loss
                loss_accumulate = 0
                for idx in range(len(outputs)):
                    focal_loss = self.loss(outputs[idx], labels[idx].to(self.device))
                    loss_accumulate += focal_loss
                    loss_tasks[idx] += focal_loss.clone().detach().cpu()

            if mode == "train":
                self.optimizer.zero_grad()
                loss_accumulate.backward()
                self.optimizer.step()

            # sum loss
            loss_sum += loss_accumulate.item()
            # print progress
            progress_bar.set_description(
                f"Loss: {loss_accumulate.item():.4f}"
            )
            # save predictions and targets
            for idx, task in enumerate(outputs):
                task_argmax = torch.argmax(task, dim = 1)
                preds[idx] += task_argmax.tolist()
                targets[idx] += labels[idx].tolist()

                # save protected labels
                # compute disparity of opportunity (eopp)
                protected_labels[idx] += protected_targets.tolist()
                pred_prob[idx] += torch.sigmoid(task.clone().detach()).to("cpu")[:,1].tolist()
                acc = (task_argmax.cpu() == labels[idx].cpu()).float()
                for i in range(2): # for each protected attribute
                    for j in range(2): # for each class
                        eopp_list[idx, i, j] += acc[(protected_targets == i) * (labels[idx] == j)].sum()
                        eopp_data_count[idx, i, j] += torch.sum((protected_targets == i) * (labels[idx] == j))

        # compute average losses
        loss_avg = loss_sum / n_iterations / self.num_attributes

        losses_to_log = []
        for idx in range(self.num_attributes):
            losses_to_log.append(
                {
                    "loss_task": loss_tasks[idx] / n_iterations,
                }
            )
            
        # compute result metrics
        performance = self._compute_accuracy(
            targets,
            preds, 
            losses_to_log,
            protected_labels,
            eopp_list,
            eopp_data_count
        )

        # print result metrics
        self._print_accuracy(
            loss_avg,
            performance
        )
        
        # save result metrics on file
        info = []
        for task in performance:
            for metric in performance[task]:
                info.append(
                    (f"{task} - {metric}: ", performance[task][metric])
                )

        self._update_result_file(
            os.path.join(
                self.result_path, 
                self.log_file
            ), 
            info
        )

        return loss_avg, info

    def _compute_accuracy(
        self, 
        targets,
        preds, 
        losses_to_log,
        protected_labels,
        eopp_list,
        eopp_data_count
    ):
        '''
        Compute the accuracy of the model.
        Args:
            preds: The predictions of the model.
            targets: The targets.
            target_names: The names of the classes.
        '''
        performance = {}

        # in this setting we have only one task
        for idx, task in enumerate(zip(preds, targets)):
            predictions = task[0]
            gt = task[1]
            f1_score = metrics.f1_score(gt, predictions, average="macro", zero_division=1)
            accuracy = metrics.accuracy_score(gt, predictions)

            # compute accuracy wrt. protected attribute
            predictions_np = np.array(predictions)
            gt_np = np.array(gt)
            protected_np = np.array(protected_labels[idx])

            protected_negative_idx = np.where(protected_np == 0)[0]
            protected_positive_idx = np.where(protected_np == 1)[0]

            predictions_protected_neg = predictions_np[protected_negative_idx]
            predictions_protected_pos = predictions_np[protected_positive_idx]
            gt_protected_neg = gt_np[protected_negative_idx]
            gt_protected_pos = gt_np[protected_positive_idx]

            acc_protected_neg = metrics.accuracy_score(gt_protected_neg, predictions_protected_neg)
            acc_protected_pos = metrics.accuracy_score(gt_protected_pos, predictions_protected_pos)

            acc_diff = acc_protected_pos - acc_protected_neg

            # compute disparity of opportunity
            eopp = eopp_list[idx] / eopp_data_count[idx]
            delta_eopp = torch.max(eopp, dim=0)[0] - torch.min(eopp, dim=0)[0]
            mean_eopp = torch.mean(delta_eopp).item()
            max_eopp = torch.max(delta_eopp).item()

            performance[self.TARGET_NAMES[idx]] = {
                "f1-score": f1_score*100,
                "accuracy": accuracy*100,
                "loss": losses_to_log[idx]["loss_task"],
                "mean_eopp": mean_eopp,
                "max_eopp": max_eopp,
                "acc_diff": acc_diff*100
            }

        return performance
    
    def _print_accuracy(self, loss, performance):
        '''
        Print the accuracy of the model.
        Args:
            loss: The loss value reached.
            performance: dict with performance values
        '''
        print("Loss average: ", loss)
        for task in performance:
            print(f"Task: {task}")
            print(f"f1-score: {performance[task]['f1-score']}")
            print(f"accuracy: {performance[task]['accuracy']}\n")
            print(f"loss: {performance[task]['loss']}\n")
            print(f"acc_diff: {performance[task]['acc_diff']}\n")
            print(f"mean_eopp: {performance[task]['mean_eopp']}\n")
            print(f"max_eopp: {performance[task]['max_eopp']}\n")
        print("\n")
    
    def _save_model(
        self,
        model,
        optimizer,
        PATH,
        epoch,
        f1_score
    ):
        '''
        Save the model.
        Args:
            model: The model to save.
            optimizer: The optimizer to save.
            PATH: The path.
            epoch: The epoch reached.
            f1_score: The f1-score achieved.
        '''
        # save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1_score': f1_score
        }, PATH)
    
    def _update_result_file(
        self,
        file_path,
        args
    ):
        '''
        Update the result file.
        Args:
            path: The path.
            file_name: The file name.
            args: The arguments to save [(arg_name, value), ... ].
        '''
        with open(file_path, "a") as f:
            for key, value in args:
                f.write(f"{key} {value}\n")
            f.write("\n")

    def print_info(
        self,
        model_name,
        n_params,
        opt
    ):
        print("\n--------------------------------Execution info--------------------------------")
        print(f" - Model: {model_name}")
        print(f" - Number of parameters: {n_params}")
        print(f" - Dataset: {opt['dataset']}")
        print(f" - N° of epochs: {opt['epochs']}")
        print(f" - Batch size: {opt['batch_size']}")
        print(f" - Learning rate: {opt['lr']}")
        print(f" - Device: {self.device}")
        print("\n")
        