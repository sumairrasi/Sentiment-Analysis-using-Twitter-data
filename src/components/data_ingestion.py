import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig,DataTokenizer,SentimentDataset
from src.components .model_trainer import ModelTrainerConfig, ModelTrainer





@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','dataset','train.csv')
    test_data_path: str = os.path.join('artifacts','dataset','test.csv')
    
    
class Dataingestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the Data ingestion method or component")
        try:
            train_df = pd.read_csv(r'data\twitter_training.csv',header=None)
            test_df = pd.read_csv(r'data\twitter_validation.csv',header=None)
            logging.info('Read the datasets as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=None)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=None)
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path,
                    
        except Exception as e:
            raise CustomException(e,sys)
            


if __name__ == '__main__':
    try:
        obj = Dataingestion()
        train_data, test_data = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_dataset,val_dataset,_=data_transformation.initiate_transformation(train_data,test_data)
        print(train_dataset)
        model_trainer = ModelTrainer()   
        print(model_trainer.initiate_model_trainer(train_dataset,val_dataset))
        logging.info("Transformation completed")
        
    except Exception as e:
        raise CustomException(e,sys)
    