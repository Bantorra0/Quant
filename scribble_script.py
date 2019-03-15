from script import *


if __name__ == '__main__':
    X, _, df_other = IO_op.read_hdf5(start="2019-01-01", end="2020-01-01",
                                     # subsample="1-0"
                                     )
    print(X.info(memory_usage="deep"))
    del X["industry"]
    # predict_dates = sorted(X.index.unique())[-20:]
    fq_cols = ["open", "high", "low", "close", "avg", "vol"]
    print(list(X.columns))
    cols = ["code"]+[col+"0" for col in fq_cols]+["amt"]+["f1mv_open"]
    df_all = pd.concat([X, df_other], axis=1)[cols]
    df_all = df_all.reset_index().set_index(["code","date"]).sort_index()
    idx = pd.IndexSlice
    codes = ['603713.SH',
             '000806.SZ',
             '600919.SH',
             '603228.SH',
             '002879.SZ',
             '300134.SZ',
             '300045.SZ']
    df = df_all.loc[idx[codes,:],:]
    for code, group in df.groupby(level="code"):
        print(group.reset_index("code").loc[-20:])

    t0 = time.time()
    r = get_return_rate(df.loc[idx[codes[0],:],fq_cols[:4]])
    print("t1:",time.time()-t0)
    t0 = time.time()
    get_return_rate2(df.loc[idx[codes[0], :], fq_cols[:4]])
    print("t2:", time.time() - t0)
    print(r)
