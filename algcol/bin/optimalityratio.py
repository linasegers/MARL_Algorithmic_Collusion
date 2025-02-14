def optimality_ratio(price, price_opponent, agent=0, environment=default, discount_rate=0.95):
    # Based on the environment, get the actions in the system
    env = Environment(*environment)
    env._create_environment()

    # Get the indices of the states that correspond to the index of the environment for the opponent limit strategy
    if agent == 0:
        idx = [env.state[x][price_opponent] for x in range(len(env.state))]
        place = 0
    else:
        idx = [env.state[price_opponent][x] for x in range(len(env.state))]
        place = 1

    # Get the corresponding profit values and discount them to get the annuity
    profit = [(env.environment[x][2][place]/(1-discount_rate)) for x in idx]

    # Actual true Q-value and maximum value (both by index)
    actual = profit[price]
    optimal = max(profit)

    return actual/optimal

# EQUILIBRIUM
# ---------------------------------------------------------------------------------------------------------------------
def equilibrium(price1, price2, environment=default, discount_rate=0.95):
    # Get the optimality ratio
    opt1 = optimality_ratio(price1, price2, agent=0, environment=environment, discount_rate=discount_rate)
    opt2 = optimality_ratio(price2, price1, agent=1, environment=environment, discount_rate=discount_rate)

    # Return whether this is in equilibrium yes or no and what the ratio is
    if opt1 == 1 and opt2 == 1:
        return 1, opt1, opt2
    else:
        return 0, opt1, opt2