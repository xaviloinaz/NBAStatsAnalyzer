import numpy as np
import pandas as pd
import csv
import pickle
import os
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
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


# Remember that the inputted first and last year are inclusive
def produce_model_data(first_year, last_year):


    labels_row = ["Explanatory Season", "Prediction Season", "Number of Players (Sample Size)", "R^2 using Model", "R^2 using just TS%", "Training Error (RMSE) using Model", "Training Error (RMSE) using just TS%"]

    for y in range(first_year, last_year):
        lbl = "Testing Error (RMSE) using Model for Seasons " + str(y) + " & " + str(y+1)
        labels_row.append(lbl)

    data_rows = []

    for year in range(first_year, last_year):

        row_list_to_write = []

        MIN_FG_EACH_SEASON = 200
        year1 = year
        year2 = year+1
        processed_merged_data_train = merged_years_setup(year1, year2, MIN_FG_EACH_SEASON, MIN_FG_EACH_SEASON)

        print(processed_merged_data_train.head(10))
        print("shape of data: ", processed_merged_data_train.shape)
        num_players = processed_merged_data_train.shape[0]
        print("Number of players: ", num_players)
        print("Number of attributes: ", processed_merged_data_train.shape[1])

        explanatory_variables_train = sm.add_constant(np.column_stack((
                                                                      processed_merged_data_train['3PAr_' + str(year1)],
                                                                      processed_merged_data_train['FTr_' + str(year1)],
                                                                      processed_merged_data_train['TS%_' + str(year1)])))

        model = sm.OLS(processed_merged_data_train['TS%_' + str(year2)], explanatory_variables_train)
        results = model.fit()
        model_rsquared = results.rsquared
        print("Summary:")
        print(results.summary())

        predictions_for_train = results.predict(explanatory_variables_train)

        explanatory_variable_basic_TS = sm.add_constant(np.column_stack((
                                                                      processed_merged_data_train['3PAr_' + str(year1)],
                                                                      processed_merged_data_train['FTr_' + str(year1)],
                                                                      processed_merged_data_train['TS%_' + str(year1)])))

        predictions_basic_TS_model = results.predict(explanatory_variable_basic_TS)

        rmse_value_training = rmse(processed_merged_data_train['TS%_' + str(year2)], predictions_for_train)
        print("RMSE for training data:", rmse_value_training)
        rmse_value_TS = rmse(processed_merged_data_train['TS%_' + str(year2)], predictions_basic_TS_model)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(processed_merged_data_train['TS%_' + str(year1)],
                                                                             processed_merged_data_train['TS%_' + str(year2)])

        print("r_value: ", r_value)
        r_squared_just_TS = r_value ** 2
        print("r_squared_value: ", r_squared_just_TS)

        row_list_to_write += [year1, year2, num_players, round(model_rsquared, 3), round(r_squared_just_TS, 3), round(rmse_value_training, 15), round(rmse_value_TS, 15)]

        for test_year in range(first_year, last_year):

            test_year_1 = test_year
            test_year_2 = test_year+1

            processed_merged_data_test = merged_years_setup(test_year_1, test_year_2, MIN_FG_EACH_SEASON,
                                                            MIN_FG_EACH_SEASON)

            explanatory_variables_test = sm.add_constant(np.column_stack((processed_merged_data_test[
                                                                              '3PAr_' + str(test_year_1)],
                                                                          processed_merged_data_test[
                                                                              'FTr_' + str(test_year_1)],
                                                                          processed_merged_data_test[
                                                                              'TS%_' + str(test_year_1)])))

            predictions_for_test = results.predict(explanatory_variables_test)

            rmse_value_test = rmse(processed_merged_data_test['TS%_' + str(test_year_2)], predictions_for_test)

            row_list_to_write.append(round(rmse_value_test, 4))


        data_rows.append(row_list_to_write)

    print(os.getcwd())
    os.chdir("/Users/xaviloinaz/Desktop/Coding/Python/BasketballStatsAnalysis/new_version")
    with open('the_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(labels_row)
        for drow in data_rows:
            writer.writerow(drow)

    print("data_rows:")
    print(data_rows)


produce_model_data(2013, 2020)




# year1 = model_explanatory_season
# year2 = model_prediction_season
# processed_merged_data_train = merged_years_setup(year1, year2, MIN_FG_EACH_SEASON, MIN_FG_EACH_SEASON)
#
# test_year_1 = 2018
# test_year_2 = 2019
# processed_merged_data_test = merged_years_setup(test_year_1, test_year_2, MIN_FG_EACH_SEASON, MIN_FG_EACH_SEASON)
#
# print(processed_merged_data_train.head(10))
# print("shape of data: ", processed_merged_data_train.shape)
# print("Number of players: ", processed_merged_data_train.shape[0])
# print("Number of attributes: ", processed_merged_data_train.shape[1])
#
# xlabel = 'TS%_' + str(year1)
# ylabel = 'TS%_' + str(year2)
#
# # explanatory_variables_train = sm.add_constant(np.column_stack((processed_merged_data_train['3PAr_' + str(year1)], processed_merged_data_train['FTr_' + str(year1)], processed_merged_data_train['2P%_totals_' + str(year1)], processed_merged_data_train['FT%_totals_' + str(year1)], processed_merged_data_train['TS%_' + str(year1)])))
# # explanatory_variables_train = sm.add_constant(np.column_stack((processed_merged_data_train['3PAr_' + str(year1)], processed_merged_data_train['FTr_' + str(year1)], processed_merged_data_train['FT%_totals_' + str(year1)], processed_merged_data_train['TS%_' + str(year1)])))
#
# explanatory_variables_train = sm.add_constant(np.column_stack((processed_merged_data_train['3PAr_' + str(year1)], processed_merged_data_train['FTr_' + str(year1)], processed_merged_data_train['TS%_' + str(year1)])))
# explanatory_variables_test = sm.add_constant(np.column_stack((processed_merged_data_test['3PAr_' + str(test_year_1)], processed_merged_data_test['FTr_' + str(test_year_1)], processed_merged_data_test['TS%_' + str(test_year_1)])))
# # explanatory_variables_train = sm.add_constant(np.column_stack((processed_merged_data_train['3PAr_' + str(year1)], processed_merged_data_train['FTr_' + str(year1)], processed_merged_data_train['TRB_per_100_poss_' + str(year1)], processed_merged_data_train['AST_per_100_poss_' + str(year1)], processed_merged_data_train['STL_per_100_poss_' + str(year1)], processed_merged_data_train['BLK_per_100_poss_' + str(year1)], processed_merged_data_train['TS%_' + str(year1)])))
# # explanatory_variables_test = sm.add_constant(np.column_stack((processed_merged_data_test['3PAr_' + str(test_year_1)], processed_merged_data_test['FTr_' + str(test_year_1)], processed_merged_data_test['TRB_per_100_poss_' + str(test_year_1)], processed_merged_data_test['AST_per_100_poss_' + str(test_year_1)], processed_merged_data_test['STL_per_100_poss_' + str(test_year_1)], processed_merged_data_test['BLK_per_100_poss_' + str(test_year_1)], processed_merged_data_test['TS%_' + str(test_year_1)])))
#
# model = sm.OLS(processed_merged_data_train['TS%_' + str(year2)], explanatory_variables_train)
# results = model.fit()
# print("Summary:")
# print(results.summary())
#
# predictions_for_train = results.predict(explanatory_variables_train)
# predictions_for_test = results.predict(explanatory_variables_test)
# print("RMSE for training data:", rmse(processed_merged_data_train['TS%_' + str(year2)], predictions_for_train))
# print("RMSE for testing data:", rmse(processed_merged_data_test['TS%_' + str(test_year_2)], predictions_for_test))
#
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(processed_merged_data_train[xlabel], processed_merged_data_train[ylabel])
# print("r_value: ", r_value)
# print("r_squared_value: ", r_value ** 2)
#
# processed_merged_data_train.plot(x=xlabel, y=ylabel, style='o')
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.show()



