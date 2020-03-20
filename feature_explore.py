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

# 生成特征
df['pct'] = df.sort_index().groupby('code')['close'].pct_change()
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