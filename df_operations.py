import pandas as pd


def str_to_none(df:pd.DataFrame):
    df[df.isnull()] = None
    for col in df.columns:
        try:
            df[(df[col]=="null") |(df[col]=="Null")
                   | (df[col]=="none") | (df[col]=="None")
                   | (df[col]=="nan") | (df[col]=="Nan")].loc[:,col] \
                = None
        except Exception as err:
            print(err)
            continue
    return df


def str_to_none_decorator(f):
    def decorator(*args,**kwargs):
        df = f(*args,**kwargs)
        return str_to_none(df)
    return decorator


def natural_join(df1:pd.DataFrame, df2:pd.DataFrame, how="inner"):
    on_cols = list(set(df1.columns) & set(df2.columns))
    return df1.merge(df2,on=on_cols, how=how)


