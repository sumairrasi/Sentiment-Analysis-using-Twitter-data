import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig,DataTokenizer,SentimentDataset
from src.components .model_trainer import ModelTrainerConfig, ModelTrainer
from src.components.data_ingestion import Dataingestion
import argparse


class TrainerPipeline:
    def __init__(self,modelconfig: ModelTrainer,transconfig: DataTransformation, dataconfig: Dataingestion):
        self.modelconfig = modelconfig
        self.transconfig = transconfig
        self.dataconfig = dataconfig

    def StartPipeline(self,train_size=None,test_size=None):
        try:
            train_data, test_data = self.dataconfig.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            size_args = {}
            if train_size is not None:
                size_args['train_size'] = train_size
            if test_size is not None:
                size_args['test_size'] = test_size
            train_dataset,val_dataset,_=self.transconfig.initiate_transformation(train_data,test_data,**size_args)
            logging.info("Transformation completed")
            print(self.modelconfig.initiate_model_trainer(train_dataset,val_dataset))
            logging.info("Model Trained completed")
        
        except Exception as e:
            raise CustomException(e, sys)




if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Process train and test sizes.')
        parser.add_argument('--train_size', type=float, help='Size of the training set')
        parser.add_argument('--test_size', type=float, help='Size of the test set')
        modelconfig = ModelTrainer()
        transconfig = DataTransformation()
        dataconfig = Dataingestion()
        trainerpipeline = TrainerPipeline(modelconfig,transconfig,dataconfig)
        args = parser.parse_args()
        trainerpipeline.StartPipeline(args.train_size, args.test_size)
        
    except Exception as e:
        raise CustomException(e,sys)