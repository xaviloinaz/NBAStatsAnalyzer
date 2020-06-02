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
    filtered_empties[filtered_empties.columns[217]] = filtered_empties[filtered_empties.columns[217]].astype(float)
    filtered_empties[filtered_empties.columns[219:233]] = filtered_empties[filtered_empties.columns[219:233]].astype(float)
    filtered_empties[filtered_empties.columns[234:238]] = filtered_empties[filtered_empties.columns[234:238]].astype(float)
    filtered_empties[filtered_empties.columns[239:243]] = filtered_empties[filtered_empties.columns[239:243]].astype(float)

    filtered_by_FG = filtered_empties[filtered_empties["FG_totals_" + str(year1)] >= minFG_year1]
    filtered_by_FG = filtered_by_FG[filtered_by_FG["FG_totals_" + str(year2)] >= minFG_year2]

    return filtered_by_FG


# Remember that the inputted first and last year are inclusive
def produce_model_data(first_year, last_year, expl_vars, file_name=None):


    labels_row = ["Explanatory Season", "Prediction Season", "Number of Players (Sample Size)", "R^2 using Model", "R^2 using just TS%", "Training Error (RMSE) using Model", "Training Error (RMSE) using just TS%"]

    for y in range(first_year, last_year):
        lbl = "Testing Error (RMSE) using Model for Seasons " + str(y) + " & " + str(y+1)
        labels_row.append(lbl)
    for y in range(first_year, last_year):
        lbl = "Testing Error (RMSE) using just TS% for Seasons " + str(y) + " & " + str(y+1)
        labels_row.append(lbl)

    labels_row.append("Average Testing Error (RMSE) using Model")
    labels_row.append("Average Testing Error (RMSE) using just TS%")

    data_rows = []

    list_of_avg_RMSE_for_model_this_year = []
    list_of_avg_RMSE_for_just_TS_model_this_year = []


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

        explanatory_variables_train = sm.add_constant(np.column_stack([processed_merged_data_train[the_stat + "_" + str(year1)] for the_stat in expl_vars]))

        model = sm.OLS(processed_merged_data_train['TS%' + "_" + str(year2)], explanatory_variables_train)
        results = model.fit()
        model_rsquared = results.rsquared
        print("Summary:")
        print(results.summary())

        predictions_for_train = results.predict(explanatory_variables_train)

        explanatory_variable_basic_TS = sm.add_constant(np.transpose(processed_merged_data_train['TS%' + "_" + str(year1)]))
        model_just_TS = sm.OLS(processed_merged_data_train['TS%'  "_" + str(year2)], explanatory_variable_basic_TS)
        results_just_TS = model_just_TS.fit()

        predictions_basic_TS_model = results_just_TS.predict(explanatory_variable_basic_TS)

        rmse_value_training = rmse(processed_merged_data_train['TS%' + "_" + str(year2)], predictions_for_train)
        print("RMSE for training data:", rmse_value_training)
        rmse_value_TS = rmse(processed_merged_data_train['TS%' + "_" + str(year2)], predictions_basic_TS_model)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(processed_merged_data_train['TS%' + "_" + str(year1)],
                                                                             processed_merged_data_train['TS%' + "_" + str(year2)])

        print("r_value: ", r_value)
        r_squared_just_TS = r_value ** 2
        print("r_squared_value: ", r_squared_just_TS)

        row_list_to_write += [year1, year2, num_players, round(model_rsquared, 3), round(r_squared_just_TS, 3), round(rmse_value_training, 4), round(rmse_value_TS, 4)]

        for test_year in range(first_year, last_year):

            test_year_1 = test_year
            test_year_2 = test_year+1

            processed_merged_data_test = merged_years_setup(test_year_1, test_year_2, MIN_FG_EACH_SEASON,
                                                            MIN_FG_EACH_SEASON)

            explanatory_variables_test = sm.add_constant(np.column_stack([processed_merged_data_test[the_stat + "_" + str(test_year_1)] for the_stat in expl_vars]))

            predictions_for_test = results.predict(explanatory_variables_test)

            rmse_value_test = rmse(processed_merged_data_test['TS%' + "_" + str(test_year_2)], predictions_for_test)

            row_list_to_write.append(round(rmse_value_test, 4))

        avg_RMSE_for_model_this_year = np.mean(row_list_to_write[-7:])
        list_of_avg_RMSE_for_model_this_year.append(avg_RMSE_for_model_this_year)

        for test_year in range(first_year, last_year):

            test_year_1 = test_year
            test_year_2 = test_year+1

            processed_merged_data_test = merged_years_setup(test_year_1, test_year_2, MIN_FG_EACH_SEASON,
                                                            MIN_FG_EACH_SEASON)

            explanatory_variables_test = sm.add_constant(np.transpose(processed_merged_data_test['TS%' + "_" + str(test_year_1)]))

            predictions_for_test = results_just_TS.predict(explanatory_variables_test)

            rmse_value_test = rmse(processed_merged_data_test['TS%' + "_" + str(test_year_2)], predictions_for_test)

            row_list_to_write.append(round(rmse_value_test, 4))

        avg_RMSE_for_just_TS_model_this_year = np.mean(row_list_to_write[-7:])
        list_of_avg_RMSE_for_just_TS_model_this_year.append(avg_RMSE_for_just_TS_model_this_year)

        row_list_to_write += [avg_RMSE_for_model_this_year, avg_RMSE_for_just_TS_model_this_year]

        data_rows.append(row_list_to_write)

    total_mean_RMSE_for_model = np.mean(list_of_avg_RMSE_for_model_this_year)
    total_mean_RMSE_for_just_TS = np.mean(list_of_avg_RMSE_for_just_TS_model_this_year)
    if file_name != None:
        print(os.getcwd())
        os.chdir("/Users/xaviloinaz/Desktop/Coding/Python/BasketballStatsAnalysis/new_version")
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(labels_row)
            for drow in data_rows:
                writer.writerow(drow)
            last_row = ["" for x in range(0,len(data_rows[0])-2)]
            last_row += [total_mean_RMSE_for_model, total_mean_RMSE_for_just_TS]
            writer.writerow(last_row)

    print("produce_model_data should have run successfully")

    # return _____need to return the data matrix so we can use it to determine
    # the column labels to iterate through and find the names of the best explanatory variables to reduce RMSE. return train or test or both??
    # total_mean_RMSE_for_model, total_mean_RMSE_for_just_TS
    return total_mean_RMSE_for_model < total_mean_RMSE_for_just_TS



def find_good_exp_variables(first_year, last_year):
    # Going to find "good" explanatory variable combinations for up to 3 variables
    possible_stats = years_stats_dict[2020].keys().tolist()

    print("Originals stats:")
    print(possible_stats)

    stats_to_remove = ['playerID', 'Player', 'Pos', 'Tm', '', u'\xa0']
    for stat in possible_stats:
        if '3P%' in stat:
            stats_to_remove.append(stat)
            print("3P% removed")

    print("stats to remove:")
    print(stats_to_remove)

    possible_stats = [x for x in possible_stats if x not in stats_to_remove]

    print("Possible stats:")
    print(possible_stats)

    exp_var_combos_1_var = list(map(lambda x: [x], possible_stats))

    exp_var_combos_2_var = []
    for stat1 in possible_stats:
        for stat2 in possible_stats:
            exp_var_combos_2_var.append([stat1, stat2])

    exp_var_combos_3_var = []
    # for stat1 in possible_stats:
    #     for stat2 in possible_stats:
    #         for stat3 in possible_stats:
    #             exp_var_combos_3_var.append([stat1, stat2, stat3])

    exp_var_combos = exp_var_combos_1_var + exp_var_combos_2_var + exp_var_combos_3_var
    good_exp_vars = []
    for exp_var_combo in exp_var_combos:
        print("Variable combination:")
        print(exp_var_combo)
        if produce_model_data(first_year, last_year, exp_var_combo):
            good_exp_vars.append(exp_var_combo)

    # # To remove unique values (for future use once you confirm the function works properly with duplicates):
    # good_exp_vars = set(list(map(lambda x: set(x), good_exp_vars)))

    return good_exp_vars




# produce_model_data(2013, 2020, ['3PAr', 'FTr', 'TS%'], "tables_produced/3PAr_FTr_TS%.csv")
# produce_model_data(2013, 2020, ['3PAr', 'FTr', 'TRB_per_100_poss', 'AST_per_100_poss', 'STL_per_100_poss', 'BLK_per_100_poss', 'TS%'], "tables_produced/3PAr_FTr_TRBper100_ASTper100_STLper100_BLKper100_TS%.csv")
# produce_model_data(2013, 2020, ['eFG%_totals'], "tables_produced/eFG%.csv")
# produce_model_data(2013, 2020, ['TS%'], "tables_produced/TS%.csv")
# produce_model_data(2013, 2020, ['TRB_per_100_poss'], "tables_produced/TRBper100.csv")
# produce_model_data(2013, 2020, ['BLK_per_100_poss'], "tables_produced/BLKper100.csv")
# produce_model_data(2013, 2020, ['eFG%_totals', 'FTr', 'FT%_totals'], "tables_produced/eFG%_FTr_FT%.csv")

good_exp_vars = find_good_exp_variables(2018, 2020)
print(good_exp_vars)

