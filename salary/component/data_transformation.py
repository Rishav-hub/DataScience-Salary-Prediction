from cgi import test
from sklearn import preprocessing
from scipy import stats
import pandas as pd
import sys,os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from salary.exception import SalaryException
from salary.logger import logging
from salary.entity.config_entity import DataTransformationConfig 
from salary.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
from salary.constant import *
from salary.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data



class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise SalaryException(e,sys) from e

    def get_data_transformer_object(self)->ColumnTransformer:
        try:

            # num_pipeline = Pipeline(steps=[
            #     ('scaler', StandardScaler())
            # ]
            # )

            # preprocessing = ColumnTransformer([
            #     ('num_pipeline', num_pipeline,),
            # ])

            # Only do standard scalar
            preprocessing = StandardScaler()
            
            return preprocessing

        except Exception as e:
            raise SalaryException(e,sys) from e   

    
    def update_job_tile(self, dataframe,original_title, updated_title):
        string_df = dataframe.job_title.str.contains(original_title)
        dataframe.loc[string_df, 'Updated_Job_Title'] = updated_title
        return dataframe

    def iqr_capping(self, df, col, factor):
    
        """_summary_ : This function is used to cap the outliers in the data using IQR method

        Returns:
            _type_: The capping function returns the capped dataframe
        """
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        upper_whisker = q3 + (factor*iqr)
        lower_whisker = q1 - (factor*iqr)

        df[col] = np.where(df[col]>upper_whisker, upper_whisker,
                    np.where(df[col]<lower_whisker, lower_whisker, df[col]))
        return df
    
    def transform_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ This function includes all the transformation on the categorical columns
            that we have done while doing analysis in the jupyter notebook

        args:
            df (pd.DataFrame): train and test data

        Raises:
            SalaryException: Any error occurs

        Returns:
            pd.DataFrame: Transformed train and test data
        """
        try:
            # 
            df.employment_type = df.employment_type.replace(['FT', 'CT', 'PT', 'FL'],['Full-time','Contract','Part-time','Freelance'])

            # New column to store the 5 categories
            # For Data Scientist
            for i in ['Research Scientist', 'Data Science Consultant', 'AI Scientist']:
                self.update_job_tile(df,i ,'Data Scientist')    

            # For Machine Learning Engineer
            for i in ['ML Engineer', 'Computer Vision Engineer', 'NLP Engineer']:
                self.update_job_tile(df,i ,'Machine Learning Engineer')

            # For Data Engineer
            for i in ['Data Architect', 'Data Science Engineer', 'Data Analytics Engineer', 'Big Data Architect', ]:
                self.update_job_tile(df,i ,'Data Engineer')
            
            # For Data Analyst
            self.update_job_tile(df,'Analytics Engineer','Data Analyst')

            # For Manager
            for i in ['Lead', 'Head', 'Director']:
                self.update_job_tile(df,i ,'Manager')
            
            # For Others
            for i in ['Computer Vision Software Engineer', 'ETL Developer', '3D Computer Vision Researcher', 'Data Specialist']:
                self.update_job_tile(df,i ,'Others')

            df.loc[df['employee_residence'].map(df['employee_residence'].value_counts(normalize=True).lt(0.03)), 'employee_residence'] = 'Others'
            df.loc[df['company_location'].map(df['company_location'].value_counts(normalize=True).lt(0.025)), 'company_location'] = 'Others'
            
            fitted_data, fitted_lambda = stats.boxcox(np.log(df['salary_in_usd']))
            df['transformed_salary_in_usd']=pd.Series(fitted_data)
            df.drop(columns=['salary','salary_in_usd','job_title','salary_currency'], inplace=True)

            # Rename the transformed salary column with salary
            df.rename(columns={'transformed_salary_in_usd':'salary'}, inplace = True)
            df.rename(columns={'Updated_Job_Title':'job_title'}, inplace = True)
            # print(df)

            # Label encode the categorical columns
            cat_features_list = [feature for feature in df.columns if df[feature].dtype == 'O']
            label_encoder = LabelEncoder()
            for cat_features in cat_features_list:
                df[cat_features]= label_encoder.fit_transform(df[cat_features])
            
            return df
        except Exception as e:
            raise SalaryException(e,sys) from e

    def transform_numerical_column(self, df:pd.DataFrame):
        """ This function includes all the transformation on the numerical columns
            that we have done while doing analysis in the jupyter notebook

        args:
            df (pd.DataFrame): train and test data

        Raises:
            SalaryException: Any error occurs

        Returns:
            pd.DataFrame: Transformed train and test data
        """
        try:
            # Apply IQR capping on the numerical columns
            numerical_columns = ['salary']
            for col in numerical_columns:
                df = self.iqr_capping(df, col, 1.5)
            return df
        except Exception as e:
            raise SalaryException(e,sys) from e

    def initiate_data_transformation(self)->DataTransformationArtifact:
        """_summary_

        Raises:
            SalaryException: _description_

        Returns:
            DataTransformationArtifact: _description_
        """
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]

            logging.info(f"Transforming Categorical features in training and test data.")
            train_df = self.transform_categorical_columns(train_df)
            test_df = self.transform_categorical_columns(test_df)

            logging.info(f"Transforming Numerical features in training and test data.")
            train_df = self.transform_numerical_column(train_df)
            test_df = self.transform_numerical_column(test_df)
            
            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            print("In data transformation")
            print(input_feature_train_df.head())
            print(input_feature_test_df.head())
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SalaryException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")
