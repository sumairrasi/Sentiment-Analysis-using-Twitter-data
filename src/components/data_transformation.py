import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
label_encoder = LabelEncoder()
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from transformers import Trainer,TrainingArguments
from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join('artifacts','preprocessor',"preprocessor.pkl")
    
    
    
@dataclass
class DataTokenizer:
    model_name = 'distilbert/distilbert-base-uncased'
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name,num_labels=4)


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
class SentimentTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings)
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    
    def initiate_transformation(self,train_path,test_path,train_size=0.3,test_size=0.2):
        try:
            train_df = pd.read_csv(train_path,names=['id','unknown','Category','Text'])
            test_df = pd.read_csv(test_path,names=['id','unknown','Category','Text'])
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)
            train_per = int(len(train_df) * train_size)
            test_per = int(len(test_df) * test_size)
            logging.info(f"The training size is: {train_per} and test size is: {test_per}")
            train_df = train_df.sample(n=train_per, random_state=42)
            test_df = test_df.sample(n=test_per, random_state=42)
            train_df['Category'] = label_encoder.fit_transform(train_df['Category'])
            test_df['Category'] = label_encoder.transform(test_df['Category'])
            train_texts = train_df['Text'].astype(str).values.tolist()
            train_labels = train_df['Category'].values.tolist()
            test_texts = test_df['Text'].astype(str).values.tolist()
            train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2,random_state=42,stratify=train_labels)
            print(train_texts)
            data_encoding = DataTokenizer()
            train_encodings = data_encoding.tokenizer(train_texts, truncation=True, padding=True,return_tensors = 'pt')
            val_encodings = data_encoding.tokenizer(val_texts, truncation=True, padding=True,return_tensors = 'pt')
            test_encodings = data_encoding.tokenizer(test_texts, truncation=True, padding=True,return_tensors = 'pt')
            train_dataset = SentimentDataset(train_encodings, train_labels)
            val_dataset = SentimentDataset(val_encodings, val_labels)
            test_dataset = SentimentTestDataset(test_encodings)
            save_object(
                
                file_path = self.data_transformation_config.preprocessor_obj_filepath,
                obj = label_encoder
            )
            
            return train_dataset,val_dataset,test_dataset
            
                    
        except Exception as e:
            raise CustomException(e,sys)



    