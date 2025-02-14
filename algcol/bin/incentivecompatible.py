
####
def incentive_compatible(profit, deviation_moment, discount_rate, method='fixed_period', endpoint=15,
                         stable=True, a1=None, a2=None):
    '''
    :param profit:
    :param deviation_moment:
    :param discount_rate:
    :param method:
    :param endpoint:
    :return:
    '''
    # Get the stable sessions only
    if stable is True and a1 is not None:
        idx = stable_sessions(a1, a2)[0]
    elif type(stable) is not bool:
        idx = stable
    else:
        try:
            idx = range(len(profit))
            warnings.warn(f'No correct measure of stability, using all sessions instead')
        except:
            # If an exception is the result, profit likely has no length, so no index needed
            pass

    # Nested function to calculate the PV of the deviation and the counter factual
    def actual_ic_check(profit, deviation_moment=deviation_moment, discount_rate=discount_rate,
                        method=method, endpoint=endpoint):
        # Change profit values to list
        profit = np.array(profit)

        stable_p = profit[deviation_moment - 1]

        # Get the period that is taken into account
        if isinstance(endpoint, int):
            profit = profit[deviation_moment:endpoint]
        else:
            profit = profit[deviation_moment:]

        # Per method, get the values of the actual play and the counterfactual that should be discounted
        if method=='fixed_period':
            values = profit
            prof = [stable_p] * len(values)
        else:
            # Alternative it to consider only the values untill the previous price is reached again, but this
            # approach thus disregards any overshooting behavior (if present)
            values = []
            p = 0

            while profit[p] < stable_p:
                values.append(profit[p])
                p = p + 1

            # Get counterfactual profit
            prof = [stable_p] * len(values)

        # Divide all the values by their discount factor
        pv_cf = np.sum([prof[x]/(1 + discount_rate)**x for x in range(len(prof))])
        pv_dev = np.sum([values[x]/(1 + discount_rate)**x for x in range(len(values))])

        # Return both PVs
        return pv_cf, pv_dev

    # If the input data is a list of lists, do this for all the sublists, and return two lists with data
    if type(profit[0]) is list or np.ndarray or pd.DataFrame:
        pv_counter = []
        pv_deviation = []
        for x in idx:
            pvc, pvd = actual_ic_check(profit=profit[x])
            pv_counter.append(pvc)
            pv_deviation.append(pvd)
        # Return the two lists
        return np.array(pv_counter), np.array(pv_deviation)
    else:
        return actual_ic_check(profit=profit)