import sqlite3
import pandas as pd
import os
from joblib import load

os.chdir("C:\\Users\\Victor\\Documents\\GeorgiaTech\\Data and Visual Analytics\\Project")
conn = sqlite3.connect('FPA_FOD_20170508.sqlite')

query = """ 
    SELECT
        time as date,
        city,
        county,
        state,
        * -- Fetch all weather metrics
    FROM AggregatedWeatherForecasts
    """

data = pd.read_sql(query, conn)



X = data.iloc[:, 10:164]
X = X.drop(columns=['sunrise_daily', 'sunset_daily'])

best_model = load('best_model.joblib')

probabilities = best_model.predict_proba(X)[:, 1]

#Chose classification threshold

result_df = pd.concat([data.iloc[:, :4], pd.Series(probabilities, name='fire_probability')], axis=1)[['state', 'city', 'county', 'date', 'fire_probability']]
result_df.to_csv('Tableu_data.csv', index = False)