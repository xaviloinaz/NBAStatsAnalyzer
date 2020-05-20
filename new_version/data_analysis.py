import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

print("Current directory:")
print(os.getcwd())
os.chdir(os.path.expanduser("~"))
print(os.getcwd())


years = list(range(1980,2021))
years_stats_dict = {}
for year_index in range(0, len(years)):
    with open('Desktop/Coding/Python/BasketballStatsAnalysis/new_version/data/player_stats_' + str(years[year_index]) + '.pkl', 'rb') as f:
        data = pickle.load(f)
        years_stats_dict[years[year_index]] = data

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.options.display.width=None

merged_2019_and_2020 = years_stats_dict[2019].merge(years_stats_dict[2020], left_on='playerID', right_on='playerID',
          suffixes=('_2019', '_2020'))



filtered_empties_2019_and_2020 = merged_2019_and_2020[merged_2019_and_2020.FGA_totals_2019 != str(0)]
filtered_empties_2019_and_2020 = filtered_empties_2019_and_2020[filtered_empties_2019_and_2020.FGA_totals_2020 != str(0)]
filtered_empties_2019_and_2020 = filtered_empties_2019_and_2020[filtered_empties_2019_and_2020.FTA_totals_2019 != str(0)]
filtered_empties_2019_and_2020 = filtered_empties_2019_and_2020[filtered_empties_2019_and_2020.FTA_totals_2020 != str(0)]

filtered_empties_2019_and_2020['FG_totals_2019'] = filtered_empties_2019_and_2020['FG_totals_2019'].astype(float)
filtered_empties_2019_and_2020['FG_totals_2020'] = filtered_empties_2019_and_2020['FG_totals_2020'].astype(float)
filtered_empties_2019_and_2020['TS%_2019'] = filtered_empties_2019_and_2020['TS%_2019'].astype(float)
filtered_empties_2019_and_2020['TS%_2020'] = filtered_empties_2019_and_2020['TS%_2020'].astype(float)
filtered_empties_2019_and_2020['FG%_totals_2019'] = filtered_empties_2019_and_2020['FG%_totals_2019'].astype(float)
filtered_empties_2019_and_2020['FG%_totals_2020'] = filtered_empties_2019_and_2020['FG%_totals_2020'].astype(float)
# filtered_empties_2019_and_2020['3P%_totals_2019'] = filtered_empties_2019_and_2020['3P%_totals_2019'].astype(float)
# filtered_empties_2019_and_2020['3P%_totals_2020'] = filtered_empties_2019_and_2020['3P%_totals_2020'].astype(float)
filtered_empties_2019_and_2020['2P%_totals_2019'] = filtered_empties_2019_and_2020['2P%_totals_2019'].astype(float)
filtered_empties_2019_and_2020['2P%_totals_2020'] = filtered_empties_2019_and_2020['2P%_totals_2020'].astype(float)
filtered_empties_2019_and_2020['FT%_totals_2019'] = filtered_empties_2019_and_2020['FT%_totals_2019'].astype(float)
filtered_empties_2019_and_2020['FT%_totals_2020'] = filtered_empties_2019_and_2020['FT%_totals_2020'].astype(float)
filtered_empties_2019_and_2020['3PAr_2019'] = filtered_empties_2019_and_2020['3PAr_2019'].astype(float)
filtered_empties_2019_and_2020['3PAr_2020'] = filtered_empties_2019_and_2020['3PAr_2020'].astype(float)
filtered_empties_2019_and_2020['FTr_2019'] = filtered_empties_2019_and_2020['FTr_2019'].astype(float)
filtered_empties_2019_and_2020['FTr_2020'] = filtered_empties_2019_and_2020['FTr_2020'].astype(float)
filtered_empties_2019_and_2020['USG%_2019'] = filtered_empties_2019_and_2020['USG%_2019'].astype(float)
filtered_empties_2019_and_2020['USG%_2020'] = filtered_empties_2019_and_2020['USG%_2020'].astype(float)
filtered_empties_2019_and_2020['eFG%_totals_2019'] = filtered_empties_2019_and_2020['eFG%_totals_2019'].astype(float)
filtered_empties_2019_and_2020['eFG%_totals_2020'] = filtered_empties_2019_and_2020['eFG%_totals_2020'].astype(float)
filtered_empties_2019_and_2020['WS/48_2019'] = filtered_empties_2019_and_2020['WS/48_2019'].astype(float)
filtered_empties_2019_and_2020['WS/48_2020'] = filtered_empties_2019_and_2020['WS/48_2020'].astype(float)
filtered_empties_2019_and_2020['OWS_2019'] = filtered_empties_2019_and_2020['OWS_2019'].astype(float)
filtered_empties_2019_and_2020['OWS_2020'] = filtered_empties_2019_and_2020['OWS_2020'].astype(float)



filtered_by_FG_2019_and_2020 = filtered_empties_2019_and_2020[filtered_empties_2019_and_2020.FG_totals_2019 >= 200]
filtered_by_FG_2019_and_2020 = filtered_by_FG_2019_and_2020[filtered_by_FG_2019_and_2020.FG_totals_2020 >= 150]

print(filtered_by_FG_2019_and_2020.head(10))
print("shape of data: ", filtered_by_FG_2019_and_2020.shape)

# xlabel = 'eFG%_totals_2019'
# ylabel = 'TS%_2019'
xlabel = 'TS%_2019'
ylabel = 'TS%_2020'


# model = sm.OLS(filtered_by_FG_2019_and_2020[ylabel], sm.add_constant(np.column_stack((filtered_by_FG_2019_and_2020['3PAr_2019'], filtered_by_FG_2019_and_2020['FTr_2019'], filtered_by_FG_2019_and_2020['2P%_totals_2019'], filtered_by_FG_2019_and_2020['FT%_totals_2019'], filtered_by_FG_2019_and_2020['TS%_2019']))))
model = sm.OLS(filtered_by_FG_2019_and_2020[ylabel], sm.add_constant(np.column_stack((filtered_by_FG_2019_and_2020['3PAr_2019'], filtered_by_FG_2019_and_2020['FTr_2019'], filtered_by_FG_2019_and_2020['TS%_2019']))))
model = sm.OLS(filtered_by_FG_2019_and_2020[ylabel], sm.add_constant(np.column_stack((filtered_by_FG_2019_and_2020['3PAr_2019'], filtered_by_FG_2019_and_2020['FTr_2019'], filtered_by_FG_2019_and_2020['2P%_totals_2019'], filtered_by_FG_2019_and_2020['3P%_totals_2019'], filtered_by_FG_2019_and_2020['FT%_totals_2019'], filtered_by_FG_2019_and_2020['TS%_2019']))))
results = model.fit()
print(results.summary())

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(filtered_by_FG_2019_and_2020[xlabel], filtered_by_FG_2019_and_2020[ylabel])
print("r_value: ", r_value)
print("r_squared_value: ", r_value ** 2)

filtered_by_FG_2019_and_2020.plot(x=xlabel, y=ylabel, style='o')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()
