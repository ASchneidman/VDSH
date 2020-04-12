import pandas as pd
import os


chunks = []
for f in os.listdir('datasets/darknet/pickled'):
    chunks.append(pd.read_pickle('datasets/darknet/pickled/' + f))

total = pd.concat(chunks)
total.to_pickle('datasets/darknet/all_appended.pkl')

'''
store=pd.HDFStore('datasets/darknet/df_all.h5')
for df in chunks:
    store.append('df',df,data_columns=df.columns, min_itemsize={'description': 9482})
#del dfs
df=store.select('df')
store.close()
os.remove('df_all.h5')
df.to_pickle('datasets/darknet/all_appended.pkl')
'''