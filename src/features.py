import pandas as pd
import numpy as np
import featuretools as ft


CSV_PATH = 'data/raw.csv'


transactions = pd.read_csv(CSV_PATH)

entityset = ft.EntitySet()
entityset = entityset.entity_from_dataframe(entity_id = 'transactions', dataframe = transactions, 
                                            index = 'ID_code')

# Perform deep feature synthesis without specifying primitives
features, feature_names = ft.dfs(entityset=entityset, target_entity='transactions', 
                                 max_depth = 2)

# No hay cambios
