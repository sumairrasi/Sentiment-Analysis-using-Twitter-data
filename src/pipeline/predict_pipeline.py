from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from dataclasses import dataclass
import pandas as pd
from src.components.model_trainer import ModelTrainerConfig
from src.components.data_transformation import DataTransformationConfig
from transformers import pipeline
from src.logger import logging
from src.exception import CustomException
import sys
from src.utils import load_object
class Predictconf:
    def __init__(self):
        self.model_path = ModelTrainerConfig()
        self.preprocess = DataTransformationConfig()
    
    

    def model_loader(self,review):
        try:
            model = DistilBertForSequenceClassification.from_pretrained(self.model_path.trained_model_file_path)
            tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path.trained_model_file_path)
            nlp = pipeline('sentiment-analysis', model=model,tokenizer=tokenizer)
            classified = nlp(review)
            return classified
        except Exception as e:
            raise CustomException(e,sys)
    
    def model_classifier(self,review):
        try:
            encoder_path = self.preprocess.preprocessor_obj_filepath
            label_encoder = load_object(encoder_path)
            classification = self.model_loader(review)
            # Define the mapping dictionary
            label_mapping = {
                'LABEL_0': 0,
                'LABEL_1': 1,
                'LABEL_2': 2,
                'LABEL_3': 3
            }

            # Iterate through the list and update the 'label' value
            for item in classification:
                item['label'] = label_mapping[item['label']]
            
            # Example encoded category value
            encoded_category = classification[0]['label']

            score = classification[0]['score']
            # Decode the encoded category using the loaded label encoder
            original_category = label_encoder.inverse_transform([encoded_category])[0]

            # Create a dictionary containing the original category and the score
            result_dict = {"result": original_category, "score": score}

            return result_dict
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == '__main__':
    pred = Predictconf()
    text ='I actually quite like the design of the ps5. It truly feels like the next generation of a console rather than just being a bulkier box with more power'
    print("Classified result is {}".format(pred.model_classifier(text)))
    



