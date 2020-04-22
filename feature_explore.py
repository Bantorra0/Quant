import pandas as pd
import numpy as np
from script import *
import itertools
import lightgbm as lgbm

pd.set_option('display.max_rows',200)

idx = pd.IndexSlice

agg_operations =['mean','median','max','min','size']

# 数据准备
start = 20130101
start_date = '-'.join([str(start)[:4],str(start)[4:6],str(start)[6:8]])
print(start_date)
# df_r = pd.read_parquet(r"database\return_8%_20%_60_20")
df_rs = pd.read_parquet(r"database\return_5%_10%_20_8_inf%").loc[idx[:,
                                                             start_date:],:].sort_index()
df_rs_sp = pd.read_parquet(r"database\return_5%_10%_20_8_15%").loc[idx[:,start_date:],:].sort_index()

df_rs1 = pd.read_parquet(r"database\return_5%_10%_20_8_inf% v1").loc[idx[:,
                                                             start_date:],:].sort_index()
df_rs_sp1 = pd.read_parquet(r"database\return_5%_10%_20_8_15% v1").loc[idx[:,
                                                                    start_date:],:].sort_index()


# df_r.sort_index(inplace=True)
print(df_r.info(memory_usage="deep"))
print(df_r.head(5))

cursor = dbop.connect_db("sqlite3").cursor()


df_d = dbop.create_df(cursor, STOCK_DAY[TABLE], start=start,
                      # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                      # where_clause="code='600350.SH'",
                      )
df_d = dp.proc_stock_d(dp.prepare_stock_d(df_d))
df_d.drop(columns=['open0','high0','low0','vol0'],inplace=True)
df_d['pct'] = df_d.sort_index().groupby('code')['close'].pct_change()*100


df_d_basic = dbop.create_df(cursor, STOCK_DAILY_BASIC[TABLE], start=start,
                            # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                            # where_clause="code='600350.SH'",
                            )
df_d_basic = dp.prepare_stock_d_basic(df_d_basic)
#
df_d_basic["pb*pe_ttm"] = df_d_basic["pb"] * df_d_basic["pe_ttm"]
df_d_basic["pb*pe"] = df_d_basic["pb"] * df_d_basic["pe"]



# index 处理
df_idx = dbop.create_df(cursor, INDEX_DAY[TABLE], start=start).sort_values(['code','date'])
df_idx['idx_pct']=df_idx.groupby('code')['close'].pct_change()
df_idx_pct = df_idx.pivot(index='date',columns='code',values='idx_pct')
df_idx_pct['idx_pct_max'] = df_idx_pct.max(axis=1)
df_idx_pct['idx_pct_mean'] = df_idx_pct.mean(axis=1)
df_idx_pct['idx_pct_median'] = df_idx_pct.median(axis=1)
df_idx_pct['idx_pct_min'] = df_idx_pct.min(axis=1)
df_idx_pct.index = pd.to_datetime(df_idx_pct.index, format='%Y%m%d')

# 拼表
df = df_d.join(df_idx_pct[['idx_pct_max','idx_pct_mean','idx_pct_median','idx_pct_min']],on='date')
df = df.join(df_d_basic)

# 生成特征
# df['pct'] = df.sort_index().groupby('code')['close'].pct_change()*100
df['win_idx'] = df['pct']>df['idx_pct_max']
df['lose_idx'] = df['pct']<df['idx_pct_min']
df['win_idx_pct'] = df['pct']-df['idx_pct_max']

days = 5
df[['win_days_{}d'.format(days),'lose_days_{}d'.format(days)]] = df[['win_idx','lose_idx']].reset_index('code').groupby('code').rolling(days).sum()
df['win_avg_pct_{}d'.format(days)] = df['win_idx_pct'].reset_index('code').groupby('code').rolling(days).mean()

days = 20
df[['win_days_{}d'.format(days),'lose_days_{}d'.format(days)]] = df[['win_idx','lose_idx']].reset_index('code').groupby('code').rolling(days).sum()
df['win_avg_pct_{}d'.format(days)] = df['win_idx_pct'].reset_index('code').groupby('code').rolling(days).mean()


filtered = df.index[(df['win_days_5d']-df['lose_days_5d']>=3) & (df['win_avg_pct_5d']<0.02) & (df['win_days_5d']-df['lose_days_5d']<=3)]
filtered = df.index[(df['win_days_5d']-df['lose_days_5d']>=3) & (df['win_avg_pct_5d']<0.01)
                    & (df['win_days_20d']-df['lose_days_20d']<=3) & (df['win_avg_pct_20d']<-0.01)]
len(filtered)/len(df)


for k in [5,10,20,30,60,120,250]:
    df['{:d}ma'.format(k)]=df.reset_index('code').groupby('code')['close'].rolling(k).mean()


# 追高明显不佳
df_r.loc[df.index[(df['close']/df['5ma']>1.05)],'r'].agg(['mean','median','size'])


df_r.loc[df.index[(df['close']/df['5ma']<0.98)
                  & (df['5ma']/df['10ma']<0.98)
                  & (df['10ma']/df['20ma']<0.98)
                  & (df['20ma']/df['30ma']<0.98)
                  & (df['30ma']/df['60ma']<0.98)],'r'].agg(['mean','median','size'])
y_result = df_r.loc[df.index[(df['close']/df['5ma']<0.98)
                             & (df['5ma']/df['10ma']<0.98)
                             & (df['10ma']/df['20ma']<0.98)
                             & (df['20ma']/df['30ma']<0.98)
                             & (df['30ma']/df['60ma']<0.98)],'r'].reset_index('code')['r'].resample('A').agg(['mean','median','size'])
q_result = df_r.loc[df.index[(df['close'] / df['5ma'] < 0.98)
                             & (df['5ma'] / df['10ma'] < 0.98)
                             & (df['10ma'] / df['20ma'] < 0.98)
                             & (df['20ma'] / df['30ma'] < 0.98)
                             & (df['30ma'] / df['60ma'] < 0.98)], 'r'].reset_index('code')['r'].resample('Q').agg(['mean', 'median', 'size'])
base_q_result = df_r.loc[df.index,'r'].reset_index('code')['r'].resample('Q').agg(['mean','median','size'])


q_result = df_r.loc[df.index[(df['close'] / df['5ma'] < 0.985)
                             & (df['5ma'] / df['10ma'] < 0.985)
                             & (df['10ma'] / df['20ma'] < 0.985)
                             & (df['20ma'] / df['30ma'] < 0.985)
                             & (df['30ma'] / df['60ma'] < 0.985)],
                    'r'].reset_index('code')['r'].resample('Q').agg([
    'mean', 'median', 'size'])


df_r.loc[df.index[(df['close'] / df['5ma'] < 0.99)
                             & (df['5ma'] / df['10ma'] < 0.99)
                             & (df['10ma'] / df['20ma'] < 0.99)
                             & (df['20ma'] / df['30ma'] < 0.99)
                             & (df['30ma'] / df['60ma'] < 0.99)
                             & (df['close']/df['60ma']<0.8)
         ],
                    'r'].reset_index('code')['r'].resample('Q').agg([
    'mean', 'median', 'size'])


q_result = df_r.loc[df.index[(df['close'] / df['5ma'] < 0.99)
                             & (df['5ma'] / df['10ma'] < 0.99)
                             & (df['10ma'] / df['20ma'] < 0.99)
                             & (df['20ma'] / df['30ma'] < 0.99)
                             & (df['30ma'] / df['60ma'] < 0.99)
                             & (df['close']/df['60ma']<0.7)]
                    ,'r'].reset_index('code')['r'].resample('Q')\
    .agg(['mean', 'median', 'size'])


df_r.loc[df.index[(df['close'] / df['5ma'] < 0.99)
                             & (df['5ma'] / df['10ma'] < 0.99)
                             & (df['10ma'] / df['20ma'] < 0.99)
                             & (df['20ma'] / df['30ma'] < 0.99)
                             & (df['30ma'] / df['60ma'] < 0.99)
                             & (df['close']/df['60ma']<0.7)
                             & (df['pct']>-0.08)]
                    ,'r'].reset_index('code')['r'].resample('M')\
    .agg(['mean', 'median','max','min', 'size'])




df_rs = pd.read_parquet(r"database\return_5%_10%_20_8")
df_rs.sort_index(inplace=True)

mask_yzdtb = ~((df['high']==df['low']) & ((df['pct'].round()<=-10) | (df['pct'].round()==-5)))
mask_strategy = ((df['close'] / df['5ma'] < 0.99)
                             & (df['5ma'] / df['10ma'] < 0.99)
                             & (df['10ma'] / df['20ma'] < 0.99)
                             & (df['20ma'] / df['30ma'] < 0.99)
                             & (df['30ma'] / df['60ma'] < 0.99)
                             & (df['close']/df['60ma']<0.7))

mask_strategy = ((df['close'] / df['5ma'] < 1)
                             & (df['5ma'] / df['10ma'] < 1)
                             & (df['10ma'] / df['20ma'] < 1)
                             & (df['20ma'] / df['30ma'] < 1)
                             & (df['30ma'] / df['60ma'] < 1)
                             & (df['close']/df['60ma']<0.7))


df_rs.loc[df.index[mask_strategy & mask_yzdtb],'r'].reset_index('code')['r'].resample('M')\
    .agg(['mean', 'median','max','min', 'size'])

df_sample = df[df['close']/df['60ma']<0.9].copy()
df_sample['cond_cnt'] = ((df_sample['close'] / df_sample['5ma'] < 1).astype('int')
                + (df_sample['5ma'] / df_sample['10ma'] < 1)
                + (df_sample['10ma'] / df_sample['20ma'] < 1)
                + (df_sample['20ma'] / df_sample['30ma'] < 1)
                + (df_sample['30ma'] / df_sample['60ma'] < 1)
                + (df_sample['60ma'] / df_sample['120ma'] < 1)
                + (df_sample['120ma'] / df_sample['250ma'] < 1)
                )

periods = [5,10,20,30,60,120,250]
# for k in periods:
#     df_sample['close/{}ma'.format(k)] = df_sample['close']/df["{}ma".format(k)]

for col1,col2 in itertools.combinations(['close']+["{}ma".format(k) for k in periods],2):
    df_sample['{}/{}'.format(col1,col2)] = df_sample[col1]/df[col2]

df_sample['upper_shadow'] = df_sample['high']/df_sample[['open','close']].max(axis=1)
df_sample['lower_shadow'] = df_sample[['open','close']].min(axis=1)/df_sample['low']
df_sample['candle_len'] = df_sample['close']/df_sample['open']

dataset = df_sample.join(df_d_basic.drop(columns='close'),how='left')
dataset = dataset.join(df_rs['r'],how='left')
dataset = dataset[~((dataset['high']==dataset['low']) & ((dataset['pct'].round()>=-10) | (dataset['pct'].round()==-5)))]
dataset = dataset[dataset.r>-0.2]
dataset = dataset[(dataset.cond_cnt>=4) & (dataset['close/60ma']<0.8)]

dataset_train = dataset.loc[idx[:,:'2019-01-01'],:]
dataset_test = dataset.loc[idx[:,'2019-01-01':],:]
print(len(dataset),len(dataset_train),len(dataset_test))

# reg = lgbm.LGBMRegressor(n_estimators=10,learning_rate=0.4,min_child_samples=len(dataset)//100,random_state=1)
n_estimators =50
callbacks = [lgbm.reset_parameter(learning_rate=lambda x:10/(n_estimators+x*2))]


reg = lgbm.LGBMRegressor(n_estimators=n_estimators,learning_rate=0.1,num_leaves=15,max_depth=8,min_child_samples=len(dataset)//100,
                         random_state=1)
lgbm.LGBMClassifier
reg.fit(dataset_train.drop(columns='r'),dataset_train['r'],
        callbacks=callbacks
        )
print(reg.score(dataset_train.drop(columns='r'),dataset_train['r']))
print(reg.score(dataset_test.drop(columns='r'),dataset_test['r']))

res = dataset_test[['r']].copy()
res['pred'] = reg.predict(dataset_test.drop(columns='r'))
res['bin'] = pd.cut(res['pred'],bins=5)

print(res.groupby('bin')['r'].agg(['mean','median','max','min','size']))
print(res.reset_index('code').groupby('bin')['r'].resample('M').agg(['mean','median','max','min','size']))

res = dataset_train[['r']].copy()
res['pred'] = reg.predict(dataset_train.drop(columns='r'))
res['bin'] = pd.cut(res['pred'],bins=5)
print(res.groupby('bin')['r'].agg(['mean','median','max','min','size']))

dataset.r.agg(['mean','median','max','min','size'])
dataset.r.resample('Q').agg(['mean','median','max','min','size'])
dataset.reset_index('code').r.resample('Q').agg(['mean','median','max','min','size'])

q_result = dataset.reset_index('code').r.resample('Q').agg(['mean','median','max','min','size'])
base_q_result = df_rs.loc[df.index,'r'].reset_index('code').r.resample('Q').agg(['mean','median','max','min','size'])

df_r.loc[df.index[(df['close'] / df['5ma'] < 0.99)
                             & (df['5ma'] / df['10ma'] < 0.99)
                             & (df['10ma'] / df['20ma'] < 0.99)
                             & (df['20ma'] / df['30ma'] < 0.99)
                             & (df['30ma'] / df['60ma'] < 0.99)
                             & (df['close']/df['60ma']<0.7)
                             & (df['pct']>-0.08)]
                    ,'r'].reset_index('code')['r'].resample('M')\
    .agg(['mean', 'median','max','min', 'size'])


df_sample_r = df_sample.join(df_rs,how='left')
df_sample_r.shape # (249357, 71)
df_sample_r = df_sample_r[~((df_sample_r['high']==df_sample_r['low']) & ((df_sample_r['pct'].round()<=-10) | (df_sample_r['pct'].round()==-5)))]
df_sample_r.shape # (248649, 71)


t_pct = 0.98
for t_pct in np.arange(0.96,1.005,0.005):
    mask_strategy = ((df_sample_r['close'] / df_sample_r['5ma'] < t_pct)
                             & (df_sample_r['5ma'] / df_sample_r['10ma'] < t_pct)
                             & (df_sample_r['10ma'] / df_sample_r['20ma'] < t_pct)
                             & (df_sample_r['20ma'] / df_sample_r['30ma'] < t_pct)
                             & (df_sample_r['30ma'] / df_sample_r['60ma'] < t_pct)
                             )
    print(t_pct)
    print(df_sample_r.loc[mask_strategy & (df_sample_r['close']/df_sample_r['60ma']<0.7),'r'].agg(['mean', 'median','max','min', 'size']))
# 结果：提升有限，size减少较多，0.985相对较好平衡。
df_sample_r.loc[(df_sample_r.cond_cnt>=5) & (df_sample_r['close']/df_sample_r['60ma']<0.7),'r'].agg(['mean', 'median','max','min', 'size'])


df_sample_r = df_sample_r.loc[idx[:,'2016-01-01':],:]
df_sample_r.loc[(df_sample_r['close']/df_sample_r['60ma']<0.65),'r'].reset_index('code')['r'].resample('Q').agg(['mean', 'median','max','min', 'size'])
df_sample_r.loc[(df_sample_r['close']/df_sample_r['60ma']<0.7),'r'].agg(['mean', 'median','max','min', 'size'])

df_sample_r2 = df_sample_r.loc[idx[:,'2016-01-01':],:]