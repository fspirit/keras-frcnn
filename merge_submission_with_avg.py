import pandas as pd

# image_filename,x0,y0,x1,y1,label,confidence

sub = pd.read_csv('./submission.csv')
print sub.info()

sub_2 = pd.read_csv('./sub_merged.csv')

# sub = pd.DataFrame(columns=['a', 'x0', 'y0'])
# sub = sub.append([dict(a=1, x0=5, y0=5), dict(a=2, x0=10, y0=10)])
#
# sub_2 = pd.DataFrame(columns=['a', 'x0', 'y0'])
# sub_2 = sub_2.append([dict(a=1, x0=7, y0=7), dict(a=2, x0=7, y0=7)])

# not_in_sub_2 = sub[~sub['image_filename'].isin(sub_2['image_filename'].values)]

# result = pd.concat([not_in_sub_2, sub_2])


sub_2['x0'] = (sub_2['x0'] + sub['x0']) / 2.0
sub_2['y0'] = (sub_2['y0'] + sub['y0']) / 2.0
sub_2['x1'] = (sub_2['x1'] + sub['x1']) / 2.0
sub_2['y1'] = (sub_2['y1'] + sub['y1']) / 2.0

# print result

sub_2.to_csv('./sub_merged_avg.csv', index=False)