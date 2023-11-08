
import pandas as pd
import os

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


from dataframe.labeling_agent import LabelingAgent

class AutoLabel(AbstractFunction):

    @setup(cacheable=False, function_type="FeatureExtraction", batchable=False)
    def setup(self, config=None):
        if config:
            self.config = config
        else:
            self.config = "spam-ham-label/config_spam_detection.json"

    @property
    def name(self) -> str:
        return "AutoLabel"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None, 5)],
            ),
        ],
        output_signatures=[
            PandasDataframe(
                columns=["response"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        task = df.iloc[0,0]
        df.drop([0,0], axis=1, inplace=True)
        # df.drop("task" , axis=1, inplace=True)
        label_df = LabelingAgent(df)
        if task=="run":
            response = label_df.label_data(self.config)
            df['class'] = response.split(',')
            df.to_csv('spam-ham-label/data/labeled_data.csv', index=False)
        elif task=="plan":
            response = label_df.check_price(self.config)
        df_dict = {"response": [str(response)]}
        
        ans_df = pd.DataFrame(df_dict)
        return pd.DataFrame(ans_df)

