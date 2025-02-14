# EXAMPLE OF SETTING COMP TABLE
import itertools
import pandas as pd
import os

###
location = 'C:/Users/linas/PycharmProjects/algocollusion/output'
filename = 'output_standard_table_exp3_2_part1.csv'

os.chdir(location)
df = pd.read_csv(filename)

types = [[x, y] for x in [0.05, 0.2, 0.35] for y in [0.1, 0.2, 0.3]]
types_in_rows = list(itertools.combinations_with_replacement(types, 2))
types_in_rows = [[[types_in_rows[x][0][0], types_in_rows[x][1][0]], [types_in_rows[x][0][1], types_in_rows[x][1][1]]] for x in range(len(types_in_rows))]

res_a0 = df['m_stand_profit_A0'].values
res_a1 = df['m_stand_profit_A0.1'].values
difs = abs(res_a0 - res_a1)


rows = []
for x in range(len(types)):
    row = []
    for y in range(len(types)):
        type_a0 = types[y]
        type_a1 = types[x]

        try:
            try:
                t = [[type_a0[0], type_a1[0]], [type_a0[1], type_a1[1]]]
                idx = types_in_rows.index(t)
                val = str(res_a0[idx])+", "+str(res_a1[idx])
            except:
                t = [[type_a1[0], type_a0[0]], [type_a1[1], type_a0[1]]]
                idx = types_in_rows.index(t)
                val = str(res_a1[idx]) + ", " + str(res_a0[idx])
        except ValueError:
            val = ''

        row.append(val)
    rows.append(row)

t = pd.DataFrame(rows)
t.index = [f'e:{types[x][0]}, a:{types[x][1]}' for x in range(len(types))]
t.columns = [f'e:{types[x][0]}, a:{types[x][1]}' for x in range(len(types))]

