import pandas as pd
import openai
from config import Config
import json



class LabelingAgent(pd.DataFrame):
    def __init__(self, df, description=None, name=None) -> None:
        super().__init__(df)

        #initialize pandas dataframe
        self.df = df        
        if len(df)>0:
            self.is_df_loaded = True
        else:
            self.is_df_loaded = False

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

    def parse_config(self, config):

        # Load the JSON configuration from a file
        with open(config, 'r') as json_file:
            config_data = json.load(json_file)

        # Extract parameters for generating the LLM prompt
        self.task_type = config_data["task_type"]
        self.labels = ", ".join(config_data["prompt"]["labels"])
        self.task_guidelines = config_data["prompt"]["task_guidelines"].replace("{labels}", self.labels)
        self.output_guidelines = config_data["prompt"]["output_guidelines"].replace("{labels}", self.labels)
        self.few_shot_examples = config_data["prompt"]["few_shot_examples"]
        self.example_template = config_data["prompt"]["example_template"]
        self.label_column = config_data["dataset"]["label_column"]
        self.examples = "Some examples with their output answers are provided below:\n"
        
        df = pd.read_csv(self.few_shot_examples)
        for index, row in df.iterrows():
            example_values = [f"{val}" for col, val in row.items() if col != self.label_column]
            example = ', '.join(example_values)
            self.examples += self.example_template.replace("{example}", example).replace("{labels}", str(row[self.label_column]))
    
    def generate_prompt_classsification_task(self):        
        # Generate the LLM prompt
        current_example = "Input: "+ str(self.df) + "Output: "
        question = "Now I want you to label the following example:\n{current_example}".replace("{current_example}", current_example)
        llm_prompt = f"{self.task_guidelines}\n{self.output_guidelines}\n{self.examples}\n{question}"
        print("LLM PROMTTT: ", llm_prompt)
        return llm_prompt
    
    def label_data(self, config):
        self.c = self.parse_config(config) #create_labelling_prompt(config)
        prompt = self.generate_prompt_classsification_task()

        answer = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=[{"role": "user", "content": prompt}])
        return answer




    
    
    


