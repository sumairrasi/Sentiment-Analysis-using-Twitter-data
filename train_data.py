import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig,DataTokenizer,SentimentDataset
from src.components .model_trainer import ModelTrainerConfig, ModelTrainer
from src.components.data_ingestion import Dataingestion
from src.pipeline.train_pipeline import TrainerPipeline
import argparse


def main(train_size, test_size):
    print("Train size:", train_size)
    print("Test size:", test_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process train and test sizes.')
    parser.add_argument('--train_size', type=float, help='Size of the training set')
    parser.add_argument('--test_size', type=float, help='Size of the test set')
    
    args = parser.parse_args()
    
    main(args.train_size, args.test_size)

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
