# Will hold all the model unitils and the model itself

import numpy as np
import pandas as pd
# import transformers
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from q3.data_processing import create_datasetdict
from q3.data_processing import create_dataloader_from_tokenized
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import DataLoader





def tokenize_function(examples, tokenizer):
    # TODO: Im not sure why, but altough i removing NaN, we are stil encountering some nan values
    # This is just a workaround that needs to be fixed
    if not examples['tweet']:
        examples['tweet'] = 'this is fake tweet'
    return tokenizer(examples["tweet"], padding="max_length", truncation=True)



class Trainer:
    def __init__(self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        model: AutoModelForSequenceClassification,
        optimizer,
        fold: int = None,
        batch_size:int = 8,
        num_epochs: int = 10,
        early_stopping: int = None,
        save_model: bool = True,
        output_name: str = None,
        only_training: bool = False,
    ):
        """
        tokenizer: Tokenizer
        model_name: model name from huggingface: https://huggingface.co/models
        fold: indicate the fold to train. If None then it will create only train
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        self.early_stopping = early_stopping
        self.save_model = save_model
        self.output_name = output_name
        self.only_training = only_training

        # Init data
        self.data = data
        # self.data.set_index('id', inplace=True)  # set the id as index so can create our prediction later.
        self.fold = fold
        
        # Init models
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        # This will create dataloaders
        # Will save the tokenized data to self.tokenized_datasets
        # And the dataloaders to self.train_dataloader and self.test_dataloader
        self.create_dataloaders()
        
        # Init optimizer and lr_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0,
            num_training_steps=self.num_epochs*len(self.train_dataloader)
            )


        self.train_loss = []
        self.test_loss = []

        self.train_acc = []
        self.test_acc = []

        
    
    def create_dataloaders(self):
        dataset = create_datasetdict(self.data, self.fold)
        self.tokenized_datasets = dataset.map(tokenize_function, fn_kwargs = {'tokenizer': self.tokenizer})
        self.train_dataloader, self.test_dataloader = create_dataloader_from_tokenized(self.tokenized_datasets, self.batch_size)

    def train(self):
        epochs_without_improvements = 0
        best_acc = 0

        for epoch in tqdm(range(self.num_epochs), desc='EPOCHS', leave=False):
            self.model.train()
            self.train_epoch()

            if not self.only_training:
                self.model.eval()
                self.eval_epoch()

                if self.test_acc[-1] > best_acc:
                    best_acc = self.test_acc[-1]
                    epochs_without_improvements = 0
                else:
                    epochs_without_improvements += 1

                    if self.early_stopping:
                        if epochs_without_improvements >= self.early_stopping:
                            break
            
            # print(f'''EPOCH: {epoch + 1}:\ntrain: loss: {self.train_loss[-1]:.3f}, acc: {self.train_acc[-1]:.3f}\ntest: loss: {self.test_loss[-1]:.3f}, acc: {self.test_acc[-1]:.3f}''')
        
        # # Save the model
        # if self.save_model:
        #     fold = self.fold if self.fold else 'all'
        #     if self.output_name:
        #         outfile = f'q3/models/{self.output_name}_fold-{fold}_{datetime.now().strftime("%d-%m-%Y_%H:%M")}'
        #     else:
        #         outfile = f'q3/models/fold-{fold}_{datetime.now().strftime("%d-%m-%Y_%H:%M")}'        
            
        #     saved_state = dict(
        #         model_state = self.model.state_dict(),
        #         test_loss = self.test_loss,
        #         test_acc = self.test_acc, 
        #     )

        #     torch.save(self.model.state_dict(), outfile)
        
    def train_epoch(self):
        acc = 0
        total = 0
        epoch_losses = []
        for batch in tqdm(self.train_dataloader, desc='train epoch', leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items() if k != 'id'}
            outputs = self.model(**batch)
            loss = outputs.loss

            epoch_losses.append(loss.item())
            loss.backward()

            y = batch['labels']
            y_pred = torch.argmax(outputs.logits, dim=-1)  # using .logits to change it to Tensor
            acc += sum(y_pred == y).item()
            total += len(y)


            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        

        acc = acc / total
        self.train_acc.append(acc)
        self.train_loss.append(np.mean(epoch_losses))

    def eval_epoch(self):
        acc = 0
        total = 0
        epoch_losses = []

        for batch in tqdm(self.test_dataloader, desc='eval epoch', leave=False):
    
            batch = {k: v.to(self.device) for k, v in batch.items() if k != 'id'}
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss

                epoch_losses.append(loss.item())

            y = batch['labels']
            y_pred = torch.argmax(outputs.logits, dim=-1)  # using .logits to change it to Tensor
            acc += sum(y_pred == y).item()
            total += len(y)
  
    
        acc = acc / total
        self.test_acc.append(acc)
        self.test_loss.append(np.mean(epoch_losses))

        
    def final_eval(self, new_data: pd.DataFrame = None):

        #TODO: process new_data to test_dataloader and then make the predictions
        
        self.model.eval()

        if new_data is not None:
            data = new_data.copy()
            self.prepare_test_dataloader(new_data)
        else:
            data = self.data.copy()

        data.set_index('id', inplace=True)
        data['y_pred'] = -1

        for batch in tqdm(self.test_dataloader, desc='eval epoch', leave=False):
            ids = batch['id'].tolist()
            batch = {k: v.to(self.device) for k, v in batch.items() if k != 'id'}

            with torch.no_grad():
                outputs = self.model(**batch)
            
            y_pred = torch.argmax(outputs.logits, dim=-1)  # using .logits to change it to Tensor
            

            data.loc[ids, 'y_pred'] = y_pred.tolist()
    
        data.reset_index(inplace=True)
        if new_data is not None:
            self.test_data = data
        else:
            self.data = data



    def prepare_test_dataloader(self, df: pd.DataFrame):
        # create datasetdict from huggingface
        datasetdict = create_datasetdict(df, test_data=True)
        datasetdict['test'] = datasetdict['train']
        del datasetdict['train']

        # tokenize the data
        tokenized_datasets = datasetdict.map(tokenize_function, fn_kwargs = {'tokenizer': self.tokenizer})
        
        # prepare to pytorch
        tokenized_datasets = tokenized_datasets.remove_columns(["tweet"])
        tokenized_datasets.set_format("torch")
        
        # create dataloader
        dataloader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size)
        self.test_dataloader = dataloader




