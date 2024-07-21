import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification
from transformers import Trainer,TrainingArguments
import dill
import os



class PredictClass():
    def __init__(self):
        self.preprocessor_obj_filepath = os.path.join('artifacts','preprocessor', "preprocessor.pkl")
        self.model_path = os.path.join('artifacts','model')
    
    def load_object(self,file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            print(e)
    
    def analysis(self,text):
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        model = DistilBertForSequenceClassification.from_pretrained(self.model_path)

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        return predicted_class_id
    
    def input_en(self,text):
        label_encoder = self.load_object(self.preprocessor_obj_filepath)
        output = self.analysis(text)
        original_category = label_encoder.inverse_transform([output])[0]

        return original_category
    
    
    
if __name__ == "__main__":
    predict = PredictClass()
    input_data = predict.input_en('he won first class')
    print(f"Predicted Class: {input_data}")