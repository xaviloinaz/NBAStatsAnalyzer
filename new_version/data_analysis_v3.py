import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

with open('./data/player_stats_' + str(2020) + '.pkl', 'rb') as f:
    player_stats_2020 = pickle.load(f)

with open('./data/player_stats_' + str(2019) + '.pkl', 'rb') as f:
    player_stats_2019 = pickle.load(f)

true_shooting_2019 = np.array(player_stats_2019['TS%'].astype(float))
true_shooting_2020 = np.array(player_stats_2020['TS%'].astype(float))


true_shooting_2019 = sm.add_constant(true_shooting_2019)
est = sm.OLS(true_shooting_2020, true_shooting_2019).fit()
print(est.summary())


# X_train = sm.add_constant(X_train)
# X_test = sm.add_constant(X_test)
# est = sm.OLS(y_train, X_train).fit()
# print(est.summary())
# y_train_pred = est.predict(X_train)
# y_test_pred = est.predict(X_test)
