import os
import sys

from salary.exception import SalaryException
from salary.util.util import load_object

import pandas as pd

"""
['work_year', 'experience_level', 'employment_type',
       'employee_residence', 'remote_ratio', 'company_location',
       'company_size', 'job_title']
"""

class SalaryData:

    def __init__(self,
                 work_year: int,
                 experience_level: str, # 
                 employment_type: str, # 
                 employee_residence: str, #
                 remote_ratio: int,
                 company_location: str, #
                 company_size: str, #
                 job_title: str, #  
                 salary: int = None
                 ):
        try:
            self.work_year = work_year
            self.experience_level = experience_level
            self.employment_type = employment_type
            self.employee_residence = employee_residence
            self.remote_ratio = remote_ratio
            self.company_location = company_location
            self.company_size = company_size
            self.job_title = job_title
            self.salary = salary
        except Exception as e:
            raise SalaryException(e, sys) from e

    def get_salary_input_data_frame(self):

        try:
            wheat_input_dict = self.get_salary_data_as_dict()
            return pd.DataFrame(wheat_input_dict)
        except Exception as e:
            raise SalaryException(e, sys) from e

    def get_salary_data_as_dict(self):
        try:
            input_data = {
                "work_year": [self.work_year],
                "experience_level": [self.experience_level],
                "employment_type": [self.employment_type],
                "employee_residence": [self.employee_residence],
                "remote_ratio": [self.remote_ratio],
                "company_location": [self.company_location],
                "company_size": [self.company_size],
                "job_title": [self.job_title],
                }
            return input_data
        except Exception as e:
            raise SalaryException(e, sys)


class SalaryPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise SalaryException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise SalaryException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            median_house_value = model.predict(X)
            return median_house_value
        except Exception as e:
            raise SalaryException(e, sys) from e