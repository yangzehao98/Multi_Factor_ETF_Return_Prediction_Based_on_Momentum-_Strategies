import pandas as pd
import pickle
from backtest.bactest import *
from model.feature_engineering import *
import itertools
from tqdm import tqdm
import copy

def main():
    # Read returns
    rets = pd.read_csv('data/returns/returns.csv', index_col='Date')
    rets.index = pd.to_datetime(rets.index)

    # Read betas
    betas = pd.read_csv('data/etf_prices/beta.csv', index_col='Date')
    betas.index = pd.to_datetime(betas.index)

    # rets = rets.drop(columns=['XLC', 'XLRE'])
    # betas = betas.drop(columns=['XLC', 'XLRE'])

    total_features = rets.columns

    etfs = [col for col in total_features if col.startswith('X')]

    with open("data/features/dataset.pkl", "rb") as file:
        data = pickle.load(file)

    # data = construct_features(rets, betas, '2023-12-31', significance_level=0.05, save=True, plot=True)

    feature_set = etfs + ['RMW', 'Mom', 'ST_Rev', '3yr_Treasury']
    # feature_set = etfs + ['SMB', 'LT_Rev', 'Gold']
    temp_data = copy.deepcopy(data)

    for etf in etfs:
        all_columns = temp_data[etf]['train'].columns
        filtered_columns = [item for item in all_columns if item.startswith(tuple(feature_set))]
        temp_train = temp_data[etf]['train'].loc[:, filtered_columns].copy()
        temp_data[etf]['train'] = temp_train
        temp_test = temp_data[etf]['test'].loc[:, filtered_columns].copy()
        temp_data[etf]['test'] = temp_test

    fit_result = fit(temp_data, save=False, verbose=True)
    trading_costs = [0.1, 0.2, 0.3]
    trading_costs = [x / 100 / 2 for x in trading_costs]# One-way trip
    trading_cost_eval_results = []
    for trading_cost in trading_costs:
        weight_schemes = ['equal', 'min_variance_optimize', ]
        weight_scheme_eval_results = []
        for weight_scheme in weight_schemes:
            rebal_eval_results = []
            for rebalance in [21, 10, 5, 1]:

                shift_eval_results = []
                if (weight_scheme == 'rolling') and rebalance != 21:
                    continue
                for shift in range(0, rebalance):
                    print('\nshift: ' + str(shift), 'Weight_scheme: ' + weight_scheme)
                    result = backtest(rets,
                                      betas,
                                      fit_result,
                                      weight_scheme=weight_scheme,
                                      shift=shift,
                                      rebalance=rebalance,
                                      period=12*6,
                                      trading_cost=trading_cost)

                    if weight_scheme != 'rolling':
                        print(f'Train Evaluation')
                        train_result = evaluate(result['train'], result['train_turnover'],
                                 data_name=f'train_shift_{weight_scheme}_{rebalance}_{shift}',
                                 weight_scheme=weight_scheme,
                                 rebalance=rebalance,)

                    print(f'\nTest Evaluation')
                    test_result = evaluate(result['test'], result['test_turnover'],
                             data_name=f'test_shift_{weight_scheme}_{rebalance}_{shift}',
                             weight_scheme=weight_scheme,
                             rebalance=rebalance,)

                    train_result = train_result.T
                    train_result.columns = [x + '_train' for x in train_result.columns]

                    test_result = test_result.T
                    test_result.columns = [x + '_test' for x in test_result.columns]

                    eval_result = pd.concat([train_result, test_result], axis=1)

                    eval_result['shift'] = shift

                    shift_eval_results.append(eval_result)

                rebal_eval_result = pd.concat(shift_eval_results, axis=0)
                rebal_eval_result['rebalance'] = rebalance
                rebal_eval_results.append(rebal_eval_result)
            weight_scheme_eval_result = pd.concat(rebal_eval_results, axis=0)
            weight_scheme_eval_result['weight_scheme'] = weight_scheme
            weight_scheme_eval_results.append(weight_scheme_eval_result)
        trading_cost_eval_result = pd.concat(weight_scheme_eval_results, axis=0)
        trading_cost_eval_result['trading_cost'] = trading_cost * 2 # Back to round trip
        trading_cost_eval_results.append(trading_cost_eval_result)
    final_eval_result = pd.concat(trading_cost_eval_results, axis=0)
    final_eval_result.to_csv('data/evaluations/weight_scheme_eval_results.csv')
    print("Saved Final Evaluation")

def feature_selection():
    # Read returns
    rets = pd.read_csv('data/returns/returns.csv', index_col='Date')
    rets.index = pd.to_datetime(rets.index)

    # Read betas
    betas = pd.read_csv('data/etf_prices/beta.csv', index_col='Date')
    betas.index = pd.to_datetime(betas.index)

    # Cut rets and betas
    rets = rets.loc[:'2023-12-31']#.drop(columns=['XLC', 'XLRE'])
    betas = betas.loc[:'2023-12-31']#.drop(columns=['XLC', 'XLRE'])

    total_features = rets.columns

    etfs = [col for col in total_features if col.startswith('X')]
    other_features = total_features.drop(etfs)

    feature_sets = []

    for size in range(0, 5):
        feature_sets.extend(itertools.combinations(other_features, size))

    feature_sets = [etfs + list(feature_set) for feature_set in feature_sets]

    # data = construct_features(rets, betas, '2021-12-31', significance_level=0.05, save=False, plot=False)

    with open("data/pickles/feature_selection_data.pkl", "wb") as file:
        pickle.dump(data, file)

    result_li = []
    for feature_set in tqdm(feature_sets):
        temp_data = copy.deepcopy(data)

        for etf in etfs:
            all_columns = temp_data[etf]['train'].columns
            filtered_columns = [item for item in all_columns if item.startswith(tuple(feature_set))]
            temp_train = temp_data[etf]['train'].loc[:, filtered_columns].copy()
            temp_data[etf]['train'] = temp_train
            temp_test = temp_data[etf]['test'].loc[:, filtered_columns].copy()
            temp_data[etf]['test'] = temp_test

        fit_result = fit(temp_data, save=False)

        sharpe_li = []
        for rebalance in [21, 10]:
            for shift in range(0, rebalance):
                result = backtest(rets, betas, fit_result, weight_scheme='equal', shift=shift, rebalance=rebalance, period=12*6)

                ptf_rets = result['test']
                annualization_factor = 252

                # Calculate cumulative returns
                cum_return = (1 + ptf_rets).cumprod()

                # Calculate annualized returns
                periods = len(cum_return)
                annual_return_ptf = cum_return['ptf'].iloc[-1] ** (annualization_factor / periods) - 1

                # Calculate annualized volatility
                volatility_ptf = ptf_rets['ptf'].std() * np.sqrt(annualization_factor)

                # Calculate Sharpe Ratio (assuming risk-free rate is 0)
                sharpe_ptf = annual_return_ptf / volatility_ptf
                sharpe_li.append(sharpe_ptf)

        result_li.append((feature_set, np.mean(sharpe_li)))
        result_li.sort(key=lambda x: x[1], reverse=True)
        print(feature_set)
        print(np.mean(sharpe_li))
        print(result_li[:10])

if __name__ == '__main__':
    main()
    # feature_selection()