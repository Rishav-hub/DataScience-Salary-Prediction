from salary.pipeline.pipeline import Pipeline
# from salary.exception import SalaryException
from salary.logger import logging
from salary.config.configuration import Configuartion
# from housing.component.data_transformation import DataTransformation
from salary.component.data_ingestion import DataIngestion
import os
def main():
    try:
        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuartion(config_file_path=config_path))
        # pipeline.run_pipeline()
        data_ingestion_artifacts = pipeline.start_data_ingestion()
        data_validation_artifacts = pipeline.start_data_validation(data_ingestion_artifacts)
        data_transformation_artifacts = pipeline.start_data_transformation(data_ingestion_artifacts, data_validation_artifacts)
        model_trainer_artifacts = pipeline.start_model_trainer(data_transformation_artifacts)
        model_evaluator_artifacts = pipeline.start_model_evaluation(data_ingestion_artifacts, data_validation_artifacts\
                                                            ,data_transformation_artifacts, model_trainer_artifacts)
        model_pusher_artifacts = pipeline.start_model_pusher(model_evaluator_artifacts)
        print(model_pusher_artifacts)

        # logging.info("main function execution completed.")
        # # data_validation_config = Configuartion().get_data_transformation_config()
        # # print(data_validation_config)
        # schema_file_path=r"D:\Project\machine_learning_project\config\schema.yaml"
        # file_path=r"D:\Project\machine_learning_project\housing\artifact\data_ingestion\2022-06-27-19-13-17\ingested_data\train\housing.csv"

        # df= DataTransformation.load_data(file_path=file_path,schema_file_path=schema_file_path)
        # print(df.columns)
        # print(df.dtypes)

    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()

