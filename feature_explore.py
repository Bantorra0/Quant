import pandas as pd
import numpy as np
from script import *

# 数据准备
df_r = pd.read_parquet(r"database\return_8%_20%_60_20")
df_r.sort_index(inplace=True)
print(df_r.info(memory_usage="deep"))
print(df_r.head(5))

cursor = dbop.connect_db("sqlite3").cursor()
start = 20130101

df_d = dbop.create_df(cursor, STOCK_DAY[TABLE], start=start,
                      # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                      # where_clause="code='600350.SH'",
                      )
df_d = dp.proc_stock_d(dp.prepare_stock_d(df_d))

df_d_basic = dbop.create_df(cursor, STOCK_DAILY_BASIC[TABLE], start=start,
                            # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                            # where_clause="code='600350.SH'",
                            )
df_d_basic = dp.prepare_stock_d_basic(df_d_basic)
#
df_d_basic["pb*pe_ttm"] = df_d_basic["pb"] * df_d_basic["pe_ttm"]
df_d_basic["pb*pe"] = df_d_basic["pb"] * df_d_basic["pe"]

df_d.drop(columns=['open0','high0','low0','vol0'],inplace=True)


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
df['pct'] = df.sort_index().groupby('code')['close'].pct_change()*100
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


for k in [5,10,20,30,60,120]:
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

mask_yzdtb = ~((df['high']==df['low']) & ((df['pct'].round()==-10) | (df['pct'].round()==-5)))
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


cond_met_cnt = ((df['close'] / df['5ma'] < 1).astype('int')
                + (df['5ma'] / df['10ma'] < 1)
                + (df['10ma'] / df['20ma'] < 1)
                + (df['20ma'] / df['30ma'] < 1)
                + (df['30ma'] / df['60ma'] < 1))
