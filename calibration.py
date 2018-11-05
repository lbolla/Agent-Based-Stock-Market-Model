import abc
import itertools

import numpy as np
from SALib.sample import latin
from tqdm import tqdm

from stockmarket import baselinemodel


def calcute_cost(
        average_price_to_earnings, upper_bound, lower_bound, observed_average):
    """Cost function for deviation from observed price to earnings ratio."""
    if lower_bound < average_price_to_earnings < upper_bound:
        return ((observed_average - average_price_to_earnings) /
                observed_average)**2
    else:
        return np.inf


def get_observed_price_to_earnings(price_to_earning):
    """Calculate the observed price to earnings."""
    p2ed = price_to_earning.describe()
    observed_average_price_to_earnings = p2ed.loc['mean']['Value']
    return observed_average_price_to_earnings


def get_price_to_earnings_window(price_to_earning):
    p2ed = price_to_earning.describe()
    return {
        'lower_min': int(p2ed.loc['min']['Value']),
        'upper_min': int(p2ed.loc['25%']['Value']),
        'lower_max': int(p2ed.loc['75%']['Value']),
        'upper_max': int(p2ed.loc['max']['Value']),
    }


def simulate_model(
        price_to_earning, initial_total_money, initial_profit, discount_rate):
    p2ew = get_price_to_earnings_window(price_to_earning)
    initial_total_money_w = (initial_total_money, initial_total_money * 1.1)
    initial_profit_w = (initial_profit, initial_profit)
    agents, firms, stocks, order_books = baselinemodel.stockMarketSimulation(
        seed=0,
        simulation_time=100,
        init_backward_simulated_time=200,
        number_of_agents=1000,
        share_chartists=0.0,
        share_mean_reversion=0.5,
        amount_of_firms=1,
        initial_total_money=initial_total_money_w,
        initial_profit=initial_profit_w,
        discount_rate=discount_rate,
        init_price_to_earnings_window=(
            (p2ew['lower_min'], p2ew['upper_min']),
            (p2ew['lower_max'], p2ew['upper_max']),
        ),
        order_expiration_time=200,
        agent_order_price_variability=(1, 1),
        agent_order_variability=1.5,
        agent_ma_short=(20, 40),
        agent_ma_long=(120, 150),
        agents_hold_thresholds=(0.9995, 1.0005),
        agent_volume_risk_aversion=0.1,
        agent_propensity_to_switch=1.1,
        firm_profit_mu=0.058,
        firm_profit_delta=0.00396825396,
        firm_profit_sigma=0.125,
        profit_announcement_working_days=20,
        printProgress=False,
        mean_reversion_memory_divider=0.5,
    )

    # 2 extract average price to earnings ratio
    average_price_to_earnings = np.mean(stocks[0].price_to_earnings_history)

    # 3 calculate costs
    observed_average_price_to_earnings = get_observed_price_to_earnings(
        price_to_earning)
    cost = calcute_cost(
        average_price_to_earnings,
        p2ew['lower_max'],
        p2ew['upper_min'],
        observed_average_price_to_earnings)

    # 4 save costs with parameter pair
    return (cost, average_price_to_earnings)


class Calibration(metaclass=abc.ABCMeta):

    def __init__(self, price_to_earning):
        self.price_to_earning = price_to_earning

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass


class FullFactorialCalibration(Calibration):

    def _get_samples(self, initial_total_money, initial_profit, discount_rate):
        return list(itertools.product(
            initial_total_money, initial_profit, discount_rate))

    def run(self, initial_total_money, initial_profit, discount_rate):

        samples = self._get_samples(
            initial_total_money, initial_profit, discount_rate)

        results = {}
        # TODO use multiprocessing
        for idx, sample in tqdm(enumerate(samples)):
            initial_total_money, initial_profit, discount_rate = sample
            results[idx] = simulate_model(
                self.price_to_earning,
                initial_total_money, initial_profit, discount_rate
            )
            if idx == 10:
                break  # TODO it takes too long!

        calibrated_parameters_location = min(results, key=results.get)
        return factors[calibrated_parameters_location]


class HypercubeCalibration(Calibration):

    def _get_samples(self, n_samples, **params):

        names = []
        bounds = []
        for name, bound in params.items():
            names.append(name)
            bounds.append(bound)

        problem = {
            "num_vars": len(params),
            "names": names,
            "bounds": bounds,
        }

        latin_hyper_cube = latin.sample(
            problem=problem, N=n_samples
        )

        return names, latin_hyper_cube

    def run(self, n_samples=500, **params):
        names, samples = self._get_samples(n_samples, **params)

        results = {}
        # for every parameter possibility:
        for idx, sample in tqdm(enumerate(samples)):

            p = dict(zip(names, sample))

            PE_low_low = p['price_to_earnings_base']
            PE_low_high = int(
                p['price_to_earnings_heterogeneity'] *
                p['price_to_earnings_base'])
            PE_high_low = PE_low_high + p['price_to_earnings_gap']
            PE_high_high = int(
                p['price_to_earnings_heterogeneity'] * PE_high_low)

            results[idx] = simulate_model(
                self.price_to_earning,
                initial_total_money, initial_profit, discount_rate
            )

        return results
