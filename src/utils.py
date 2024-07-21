
import os
import sys

import numpy as np
import pandas as pd
from src.exception import CustomException
import dill


def save_object(file_path, obj):
    try:
        dir_path =os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        print(e)

preprocessor_obj_filepath = os.path.join('artifacts','preprocessor', "preprocessor.pkl")
