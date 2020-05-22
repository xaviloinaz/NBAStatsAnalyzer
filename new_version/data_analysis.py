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




def merged_years_setup(year1, year2, minFG_year1, minFG_year2): # The input years are ints
    merged = years_stats_dict[year1].merge(years_stats_dict[year2], left_on='playerID', right_on='playerID',
          suffixes=('_' + str(year1), '_' + str(year2)))
    filtered_empties = merged[merged["FGA_totals_" + str(year1)] != str(0)]
    filtered_empties = filtered_empties[filtered_empties["FGA_totals_" + str(year2)] != str(0)]
    filtered_empties = filtered_empties[filtered_empties["FTA_totals_" + str(year1)] != str(0)]
    filtered_empties = filtered_empties[filtered_empties["FTA_totals_" + str(year2)] != str(0)]
    filtered_empties = filtered_empties[filtered_empties["2PA_totals_" + str(year1)] != str(0)]
    filtered_empties = filtered_empties[filtered_empties["2PA_totals_" + str(year2)] != str(0)]

    # # Useful for debugging purposes:
    # i = 0
    # for col_name in filtered_empties.columns:
    #     print(str(i) + ":")
    #     print(col_name)
    #     i += 1
    # print(filtered_empties.head(10))

    filtered_empties[filtered_empties.columns[0:6]] = filtered_empties[filtered_empties.columns[0:6]].astype(float)
    filtered_empties[filtered_empties.columns[7:29]] = filtered_empties[filtered_empties.columns[7:29]].astype(float)
    filtered_empties[filtered_empties.columns[30:52]] = filtered_empties[filtered_empties.columns[30:52]].astype(float)
    filtered_empties[filtered_empties.columns[53:74]] = filtered_empties[filtered_empties.columns[53:74]].astype(float)
    filtered_empties[filtered_empties.columns[75:90]] = filtered_empties[filtered_empties.columns[75:90]].astype(float)
    filtered_empties[filtered_empties.columns[91:93]] = filtered_empties[filtered_empties.columns[91:93]].astype(float)
    filtered_empties[filtered_empties.columns[98:112]] = filtered_empties[filtered_empties.columns[98:112]].astype(float)
    filtered_empties[filtered_empties.columns[113:117]] = filtered_empties[filtered_empties.columns[113:117]].astype(float)
    filtered_empties[filtered_empties.columns[118:128]] = filtered_empties[filtered_empties.columns[118:128]].astype(float)
    filtered_empties[filtered_empties.columns[129:151]] = filtered_empties[filtered_empties.columns[129:151]].astype(float)
    filtered_empties[filtered_empties.columns[152:174]] = filtered_empties[filtered_empties.columns[152:174]].astype(float)
    filtered_empties[filtered_empties.columns[175:196]] = filtered_empties[filtered_empties.columns[175:196]].astype(float)
    filtered_empties[filtered_empties.columns[197:212]] = filtered_empties[filtered_empties.columns[197:212]].astype(float)
    filtered_empties[filtered_empties.columns[213:215]] = filtered_empties[filtered_empties.columns[213:215]].astype(float)
    filtered_empties[filtered_empties.columns[219:233]] = filtered_empties[filtered_empties.columns[219:233]].astype(float)
    filtered_empties[filtered_empties.columns[234:238]] = filtered_empties[filtered_empties.columns[234:238]].astype(float)
    filtered_empties[filtered_empties.columns[239:243]] = filtered_empties[filtered_empties.columns[239:243]].astype(float)

    filtered_by_FG = filtered_empties[filtered_empties["FG_totals_" + str(year1)] >= minFG_year1]
    filtered_by_FG = filtered_by_FG[filtered_by_FG["FG_totals_" + str(year2)] >= minFG_year2]

    return filtered_by_FG


#
# merged_2019_and_2020 = years_stats_dict[2019].merge(years_stats_dict[2020], left_on='playerID', right_on='playerID',
#           suffixes=('_2019', '_2020'))
#
# filtered_empties_2019_and_2020 = merged_2019_and_2020[merged_2019_and_2020.FGA_totals_2019 != str(0)]
# filtered_empties_2019_and_2020 = filtered_empties_2019_and_2020[filtered_empties_2019_and_2020.FGA_totals_2020 != str(0)]
# filtered_empties_2019_and_2020 = filtered_empties_2019_and_2020[filtered_empties_2019_and_2020.FTA_totals_2019 != str(0)]
# filtered_empties_2019_and_2020 = filtered_empties_2019_and_2020[filtered_empties_2019_and_2020.FTA_totals_2020 != str(0)]
#
# filtered_empties_2019_and_2020['FG_totals_2019'] = filtered_empties_2019_and_2020['FG_totals_2019'].astype(float)
# filtered_empties_2019_and_2020['FG_totals_2020'] = filtered_empties_2019_and_2020['FG_totals_2020'].astype(float)
# filtered_empties_2019_and_2020['TS%_2019'] = filtered_empties_2019_and_2020['TS%_2019'].astype(float)
# filtered_empties_2019_and_2020['TS%_2020'] = filtered_empties_2019_and_2020['TS%_2020'].astype(float)
# filtered_empties_2019_and_2020['FG%_totals_2019'] = filtered_empties_2019_and_2020['FG%_totals_2019'].astype(float)
# filtered_empties_2019_and_2020['FG%_totals_2020'] = filtered_empties_2019_and_2020['FG%_totals_2020'].astype(float)
# # filtered_empties_2019_and_2020['3P%_totals_2019'] = filtered_empties_2019_and_2020['3P%_totals_2019'].astype(float)
# # filtered_empties_2019_and_2020['3P%_totals_2020'] = filtered_empties_2019_and_2020['3P%_totals_2020'].astype(float)
# filtered_empties_2019_and_2020['2P%_totals_2019'] = filtered_empties_2019_and_2020['2P%_totals_2019'].astype(float)
# filtered_empties_2019_and_2020['2P%_totals_2020'] = filtered_empties_2019_and_2020['2P%_totals_2020'].astype(float)
# filtered_empties_2019_and_2020['FT%_totals_2019'] = filtered_empties_2019_and_2020['FT%_totals_2019'].astype(float)
# filtered_empties_2019_and_2020['FT%_totals_2020'] = filtered_empties_2019_and_2020['FT%_totals_2020'].astype(float)
# filtered_empties_2019_and_2020['3PAr_2019'] = filtered_empties_2019_and_2020['3PAr_2019'].astype(float)
# filtered_empties_2019_and_2020['3PAr_2020'] = filtered_empties_2019_and_2020['3PAr_2020'].astype(float)
# filtered_empties_2019_and_2020['FTr_2019'] = filtered_empties_2019_and_2020['FTr_2019'].astype(float)
# filtered_empties_2019_and_2020['FTr_2020'] = filtered_empties_2019_and_2020['FTr_2020'].astype(float)
# filtered_empties_2019_and_2020['USG%_2019'] = filtered_empties_2019_and_2020['USG%_2019'].astype(float)
# filtered_empties_2019_and_2020['USG%_2020'] = filtered_empties_2019_and_2020['USG%_2020'].astype(float)
# filtered_empties_2019_and_2020['eFG%_totals_2019'] = filtered_empties_2019_and_2020['eFG%_totals_2019'].astype(float)
# filtered_empties_2019_and_2020['eFG%_totals_2020'] = filtered_empties_2019_and_2020['eFG%_totals_2020'].astype(float)
# filtered_empties_2019_and_2020['WS/48_2019'] = filtered_empties_2019_and_2020['WS/48_2019'].astype(float)
# filtered_empties_2019_and_2020['WS/48_2020'] = filtered_empties_2019_and_2020['WS/48_2020'].astype(float)
# filtered_empties_2019_and_2020['OWS_2019'] = filtered_empties_2019_and_2020['OWS_2019'].astype(float)
# filtered_empties_2019_and_2020['OWS_2020'] = filtered_empties_2019_and_2020['OWS_2020'].astype(float)
#
#
# filtered_by_FG_2019_and_2020 = filtered_empties_2019_and_2020[filtered_empties_2019_and_2020.FG_totals_2019 >= 200]
# filtered_by_FG_2019_and_2020 = filtered_by_FG_2019_and_2020[filtered_by_FG_2019_and_2020.FG_totals_2020 >= 150]


year1 = 2018
year2 = 2019
processed_merged_data = merged_years_setup(year1, year2, 200, 200)

print(processed_merged_data.head(10))
print("shape of data: ", processed_merged_data.shape)

# xlabel = 'FT%_totals_' + str(year2)
xlabel = 'TS%_' + str(year1)
ylabel = 'TS%_' + str(year2)

# model = sm.OLS(processed_merged_data['TS%_' + str(year2)], sm.add_constant(np.column_stack((processed_merged_data['3PAr_' + str(year1)], processed_merged_data['FTr_' + str(year1)], processed_merged_data['2P%_totals_' + str(year1)], processed_merged_data['FT%_totals_' + str(year1)], processed_merged_data['TS%_' + str(year1)]))))
# model = sm.OLS(processed_merged_data['TS%_' + str(year2)], sm.add_constant(np.column_stack((processed_merged_data['3PAr_' + str(year1)], processed_merged_data['FTr_' + str(year1)], processed_merged_data['TS%_' + str(year1)]))))
# model = sm.OLS(processed_merged_data['TS%_' + str(year2)], sm.add_constant(np.column_stack((processed_merged_data['3PAr_' + str(year1)], processed_merged_data['FTr_' + str(year1)], processed_merged_data['FT%_totals_' + str(year1)], processed_merged_data['TS%_' + str(year1)]))))
model = sm.OLS(processed_merged_data['TS%_' + str(year2)], sm.add_constant(np.column_stack((processed_merged_data['3PAr_' + str(year1)], processed_merged_data['FTr_' + str(year1)], processed_merged_data['TRB_per_100_poss_' + str(year1)], processed_merged_data['AST_per_100_poss_' + str(year1)], processed_merged_data['STL_per_100_poss_' + str(year1)], processed_merged_data['BLK_per_100_poss_' + str(year1)], processed_merged_data['TS%_' + str(year1)]))))
results = model.fit()
print(results.summary())

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(processed_merged_data[xlabel], processed_merged_data[ylabel])
print("r_value: ", r_value)
print("r_squared_value: ", r_value ** 2)

processed_merged_data.plot(x=xlabel, y=ylabel, style='o')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()

