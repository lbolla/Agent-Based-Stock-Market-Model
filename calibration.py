import numpy as np

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
    # calculate the observed price to earnings
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


def simulate_model(price_to_earning, parameters):
    p2ew = get_price_to_earnings_window(price_to_earning)
    agents, firms, stocks, order_books = baselinemodel.stockMarketSimulation(
        seed=0,
        simulation_time=100,
        init_backward_simulated_time=200,
        number_of_agents=1000,
        share_chartists=0.0,
        share_mean_reversion=0.5,
        amount_of_firms=1,
        initial_total_money=(parameters[0], parameters[0]*1.1),
        initial_profit=(parameters[1], parameters[1]),
        discount_rate=parameters[2],
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
