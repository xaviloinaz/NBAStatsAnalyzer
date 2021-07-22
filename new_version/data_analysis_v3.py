import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score


years = list(range(1998,2021))
years_stats_dict = {}
for year_index in range(0, len(years)):
    with open('./data/player_stats_' + str(years[year_index]) + '.pkl', 'rb') as f:
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

    # filtered_empties[filtered_empties.columns[0:6]] = filtered_empties[filtered_empties.columns[0:6]].astype(float)
    # filtered_empties[filtered_empties.columns[7:29]] = filtered_empties[filtered_empties.columns[7:29]].astype(float)
    # filtered_empties[filtered_empties.columns[30:52]] = filtered_empties[filtered_empties.columns[30:52]].astype(float)
    # filtered_empties[filtered_empties.columns[53:74]] = filtered_empties[filtered_empties.columns[53:74]].astype(float)
    # filtered_empties[filtered_empties.columns[75:90]] = filtered_empties[filtered_empties.columns[75:90]].astype(float)
    # filtered_empties[filtered_empties.columns[91:93]] = filtered_empties[filtered_empties.columns[91:93]].astype(float)
    # filtered_empties[filtered_empties.columns[96]] = filtered_empties[filtered_empties.columns[96]].astype(float)
    # filtered_empties[filtered_empties.columns[98:112]] = filtered_empties[filtered_empties.columns[98:112]].astype(float)
    # filtered_empties[filtered_empties.columns[113:117]] = filtered_empties[filtered_empties.columns[113:117]].astype(float)
    # filtered_empties[filtered_empties.columns[118:128]] = filtered_empties[filtered_empties.columns[118:128]].astype(float)
    # filtered_empties[filtered_empties.columns[129:151]] = filtered_empties[filtered_empties.columns[129:151]].astype(float)
    # filtered_empties[filtered_empties.columns[152:174]] = filtered_empties[filtered_empties.columns[152:174]].astype(float)
    # filtered_empties[filtered_empties.columns[175:196]] = filtered_empties[filtered_empties.columns[175:196]].astype(float)
    # filtered_empties[filtered_empties.columns[197:212]] = filtered_empties[filtered_empties.columns[197:212]].astype(float)
    # filtered_empties[filtered_empties.columns[213:215]] = filtered_empties[filtered_empties.columns[213:215]].astype(float)
    # filtered_empties[filtered_empties.columns[217]] = filtered_empties[filtered_empties.columns[217]].astype(float)
    # filtered_empties[filtered_empties.columns[219:233]] = filtered_empties[filtered_empties.columns[219:233]].astype(float)
    # filtered_empties[filtered_empties.columns[234:238]] = filtered_empties[filtered_empties.columns[234:238]].astype(float)
    # filtered_empties[filtered_empties.columns[239:243]] = filtered_empties[filtered_empties.columns[239:243]].astype(float)

    filtered_by_FG = filtered_empties[filtered_empties["FG_totals_" + str(year1)].astype(int) >= minFG_year1]
    filtered_by_FG = filtered_by_FG[filtered_by_FG["FG_totals_" + str(year2)].astype(int) >= minFG_year2]

    return filtered_by_FG

merged_years_2019_2020 = merged_years_setup(2019,2020,200,200)

print(merged_years_2019_2020)

true_shooting_2019 = np.array(merged_years_2019_2020['TS%_2019'].astype(float))
true_shooting_2020 = np.array(merged_years_2019_2020['TS%_2020'].astype(float))

true_shooting_2019 = sm.add_constant(true_shooting_2019)
est = sm.OLS(true_shooting_2020, true_shooting_2019).fit()
print("Regression just using previous year's true shooting percentage:")
print("R-squared:", est.rsquared)
print("R:", np.sqrt(est.rsquared))
print("R-squared without OLS (like just R^2 year-to-year of true shooting percentages:", r2_score(true_shooting_2020, true_shooting_2019[:,1]))
print("R without OLS (like just R^2 year-to-year of true shooting percentages:", np.sqrt(r2_score(true_shooting_2020, true_shooting_2019[:,1])))
# print(est.summary())

print()

shot_0_to_3_2019 = np.expand_dims(np.array(merged_years_2019_2020['0-3_fga%_2019'].astype(float)),axis=1)
shot_3_to_10_2019 = np.expand_dims(np.array(merged_years_2019_2020['3-10_fga%_2019'].astype(float)),axis=1)
shot_10_to_16_2019 = np.expand_dims(np.array(merged_years_2019_2020['10-16_fga%_2019'].astype(float)),axis=1)
shot_16_to_3pt_2019 = np.expand_dims(np.array(merged_years_2019_2020['16-3P_fga%_2019'].astype(float)),axis=1)
shot_3pt_2019 = np.expand_dims(np.array(merged_years_2019_2020['3P_fga%_2019'].astype(float)),axis=1)


shooting_data_2019 = np.hstack((shot_0_to_3_2019,shot_3_to_10_2019,shot_10_to_16_2019,shot_16_to_3pt_2019,shot_3pt_2019))
shooting_data_2019 = sm.add_constant(shooting_data_2019)
est = sm.OLS(true_shooting_2020, shooting_data_2019).fit()
print("Regression just using basic shot range information:")
print("R-squared:", est.rsquared)
print("R:", np.sqrt(est.rsquared))
# print(est.summary())

all_shooting_data_2019 = np.hstack((true_shooting_2019,shot_0_to_3_2019,shot_3_to_10_2019,shot_10_to_16_2019,shot_16_to_3pt_2019,shot_3pt_2019))
all_shooting_data_2019 = sm.add_constant(all_shooting_data_2019)
est = sm.OLS(true_shooting_2020, all_shooting_data_2019).fit()
print("Multiple linear regression using previous year's true shooting percentage and shot range information:")
print("R-squared:", est.rsquared)
print("R:", np.sqrt(est.rsquared))
# print(est.summary())

print()



shot_FG_percent_0_to_3_2019 = np.expand_dims(np.array(merged_years_2019_2020['0-3_fg%_2019'].astype(float)),axis=1)
shot_FG_percent_3_to_10_2019 = np.expand_dims(np.array(merged_years_2019_2020['3-10_fg%_2019'].astype(float)),axis=1)
shot_FG_percent_10_to_16_2019 = np.expand_dims(np.array(merged_years_2019_2020['10-16_fg%_2019'].astype(float)),axis=1)
shot_FG_percent_16_to_3pt_2019 = np.expand_dims(np.array(merged_years_2019_2020['16-3P_fg%_2019'].astype(float)),axis=1)
shot_FG_percent_3pt_2019 = np.expand_dims(np.array(merged_years_2019_2020['3P_fg%_2019'].astype(float)),axis=1)



shooting_data_2019 = np.hstack((shot_0_to_3_2019,shot_3_to_10_2019,shot_10_to_16_2019,shot_16_to_3pt_2019,shot_3pt_2019,shot_FG_percent_0_to_3_2019,shot_FG_percent_3_to_10_2019,shot_FG_percent_10_to_16_2019,shot_FG_percent_16_to_3pt_2019,shot_FG_percent_3pt_2019))
shooting_data_2019 = sm.add_constant(shooting_data_2019)
est = sm.OLS(true_shooting_2020, shooting_data_2019).fit()
print("Regression using basic shot range information and field goal percentage at ranges:")
print("R-squared:", est.rsquared)
print("R:", np.sqrt(est.rsquared))
# print(est.summary())

all_shooting_data_2019 = np.hstack((true_shooting_2019,shot_0_to_3_2019,shot_3_to_10_2019,shot_10_to_16_2019,shot_16_to_3pt_2019,shot_3pt_2019,shot_FG_percent_0_to_3_2019,shot_FG_percent_3_to_10_2019,shot_FG_percent_10_to_16_2019,shot_FG_percent_16_to_3pt_2019,shot_FG_percent_3pt_2019))
all_shooting_data_2019 = sm.add_constant(all_shooting_data_2019)
est = sm.OLS(true_shooting_2020, all_shooting_data_2019).fit()
print("Multiple linear regression using previous year's true shooting percentage as well as shot range information and field goal percentage at ranges:")
print("R-squared:", est.rsquared)
print("R:", np.sqrt(est.rsquared))
# print(est.summary())



# X_train = sm.add_constant(X_train)
# X_test = sm.add_constant(X_test)
# est = sm.OLS(y_train, X_train).fit()
# print(est.summary())
# y_train_pred = est.predict(X_train)
# y_test_pred = est.predict(X_test)
