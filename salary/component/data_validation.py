
from salary.logger import logging
from salary.exception import SalaryException
from salary.entity.config_entity import DataValidationConfig
from salary.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from salary.util.util import read_yaml_file

import os,sys
import pandas  as pd



class DataValidation:
    

    def __init__(self, data_validation_config:DataValidationConfig,
        data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*30}Data Valdaition log started.{'<<'*30} \n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SalaryException(e,sys) from e


    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df,test_df
        except Exception as e:
            raise SalaryException(e,sys) from e


    def is_train_test_file_exists(self)->bool:
        try:
            logging.info("Checking if training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available =  is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")
            
            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message=f"Training file: {training_file} or Testing file: {testing_file}" \
                    "is not present"
                raise Exception(message)

            return is_available
        except Exception as e:
            raise SalaryException(e,sys) from e

    
    def validate_dataset_schema(self)->bool:
        try:
            # Get the train and test dataframes
            train_df,test_df = self.get_train_and_test_df()
            # Get the schema file path
            schema_file_path = self.data_validation_config.schema_file_path
            # Read the schema file
            schema_df = read_yaml_file(schema_file_path)
            
            # First set the validation status to False
            validation_status = True

            # Validation for train data
            if len(train_df.columns) != len(schema_df['columns']):
                validation_status = False
                return validation_status

            for i in range(len(schema_df['columns'])):
                if list(schema_df['columns'].keys())[i] == train_df.columns[i]:
                    validation_status = True
                else:
                    validation_status = False
                    return validation_status
            
            # Validation for test data
            # Change the name of the by removing the punctuations
            # test_df.columns = test_df.columns.str.replace('.', '_', regex=True)

            if len(test_df.columns) != len(schema_df['columns']):
                validation_status = False
                return validation_status

            for i in range(len(schema_df['columns'])):
                if list(schema_df['columns'].keys())[i] == test_df.columns[i]:
                    validation_status = True
                else:
                    validation_status = False
                    return validation_status
            return validation_status
        except Exception as e:
            raise SalaryException(e,sys) from e

    def initiate_data_validation(self)->DataValidationArtifact :
        try:
            self.is_train_test_file_exists()
            validation_status = self.validate_dataset_schema()
            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                is_validated=validation_status,
                message="Data Validation performed successully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SalaryException(e,sys) from e


    def __del__(self):
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")
        



