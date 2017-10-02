import pandas as pd


sub = pd.read_csv('./submission.csv')
sub_2 = pd.read_csv('./submission_2.csv')

# df0 = pd.DataFrame(columns=['a', 'b'])
# df0 = df0.append([dict(a=1, b='eee'), dict(a=2, b='eeeaads')])
#
# df1 = pd.DataFrame(columns=['a', 'b'])
# df1 = df1.append([dict(a=1, b='ffff')])

df0 = sub[~sub['a'].isin(sub_2['a'].values)]

result = pd.concat([sub, sub_2])

print result.info()

result.to_csv('./sub_merged.csv', index=False)


