import pandas as pd
import openai
from config import Config
import re
import os


class AIDataFrame(pd.DataFrame):
    def __init__(self, df, config=None, description=None, name=None) -> None:
        super().__init__(df)

        #initialize pandas dataframe
        self.pd_df = df
        self.config = Config()
        
        if len(df)>0:
            self.is_df_loaded = True
        else:
            self.is_df_loaded = False
        
        #set the config
        if config:
            self.config = config

    @property
    def col_count(self):
        if self.is_df_loaded:
            return len(list(self.pd_df.columns))
        
    @property
    def row_count(self):
        if self.is_df_loaded:
            return len(self.pd_df)
        
    @property
    def sample_head_csv(self):
        if self.is_df_loaded:
            return self.pd_df.head(5).to_csv()
        
    
    @property
    def metadata(self):
        return self.pd_df.info()
    
    def to_csv(self, file_path):
        self.pd_df.to_csv(file_path)
    
        
    def initialize_middleware(self):
        """ Initializes openai api with the openai key and model """
        open_ai_key = self.config.get_open_ai_key()
        openai.api_key = open_ai_key
        self.openai_model = "gpt-3.5-turbo"
        return

    def create_data_cleaning_prompt(self, clean_query: str):
        prompt = f"I need you to write a python3.8 program for the following dataframe. \
            You are given the following pandas dataframe. \
            The dataframe has {self.col_count} columns. The columns are {list(self.columns)}. \
            The first 2 rows of data in the csv format are {self.iloc[0:2].to_csv()} .\
            Give me the python code to perform the following data cleaning: {clean_query}.\
            Write this code in a function named 'pandas_clean_function' and it should take the pandas dataframe as input. \
            Do not create a new dataframe. assume that it is given as input to the function.\
            The output should be the dataframe after the cleaning are done.\
            Add the required imports for the function. \
            Do not add any code for example usage to execute the function. Write only the function code.\
            The response should have only the python code and no additional text. \
            I repeat.. give the python code only for the function. NO ADDITIONAL CODE."
        return prompt


    def execute_python(self, python_code: str, type: str):
        """
         A function to execute the python code and return result. 
         
         Args
         python_code - the python code to be executed. It is in string format.
         type - type of function to be executed. query | plot | manipulation
         
         Returns
         Return the result of the execution.
        """

        if type == "data_cleaning":
            with open("tmp.py", "w+") as file:
                file.write(python_code)
            
            from tmp import pandas_clean_function
            output  = pandas_clean_function(self.pd_df)

            os.remove("tmp.py")
            return output
    
    def label_data(self, instructions):
        prompt = self.create_data_cleaning_prompt(clean_instructions)
        # get instrcutions from the config
        #get model

        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=[{"role": "user", "content": prompt}])
        
        # python_code = completion.choices[0].message.content
        # answer = self.execute_python(python_code, "data_cleaning")
        return answer




    
    
    


