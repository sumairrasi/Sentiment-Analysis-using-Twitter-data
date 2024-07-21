from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from transformers import Trainer,TrainingArguments
import os
os.environ['WANDB_DISABLED'] = 'true'
import sys
from src.logger import logging
from src.exception import CustomException


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    #recall = recall_score(y_true=labels, y_pred=pred)
    #precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred, average='weighted')

    return {"accuracy": accuracy,"f1_score":f1}

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model')
    output_path = os.path.join('artifacts','results')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model_name = 'distilbert/distilbert-base-uncased'
    
    def initiate_model_trainer(self,train_data,val_data):
        try:
            training_args = TrainingArguments(
                output_dir=self.model_trainer_config.output_path,          # output directory
                evaluation_strategy="steps",
                num_train_epochs=2,              # total number of training epochs
                per_device_train_batch_size=4,  # batch size per device during training
                per_device_eval_batch_size=2,   # batch size for evaluation
                warmup_steps=500,                # number of warmup steps for learning rate scheduler
                weight_decay=0.01,               # strength of weight decay
                logging_dir='./logs4',            # directory for storing logs
                #logging_steps=10,
                load_best_model_at_end=True,
            )
            model = DistilBertForSequenceClassification.from_pretrained(self.model_name,num_labels=4)
            tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name,num_labels=4)
            trainer = Trainer(
                model=model,# the instantiated 🤗 Transformers model to be trained
                args=training_args, # training arguments, defined above
                train_dataset=train_data,# training dataset
                eval_dataset=val_data , # evaluation dataset
                compute_metrics=compute_metrics,
            )

            trainer.train()
            
            trainer.save_model(self.model_trainer_config.trained_model_file_path)
            tokenizer.save_pretrained(self.model_trainer_config.trained_model_file_path)
            
            evaluation = trainer.evaluate()

            return evaluation
        
        except Exception as e:
            raise CustomException(e,sys)