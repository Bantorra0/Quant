import pandas as pd
import datetime
import pickle
import os

import ml_model


def get_interval_dates(slice_length=6):
    if slice_length is None:
        slice_length=6
    return ["{0:02d}01".format(1 + i * slice_length) for i in
             range(12 // slice_length)]


def get_interval_mon_day(mon, day,slice_length=None,dates=None):
    if dates is None:
        dates = get_interval_dates(slice_length)
    l = [(i,date) for i,date in enumerate(dates) if mon + day < date]
    if len(l) > 0:
        interval_end = l[0][1]
        interval_start = dates[l[0][0] - 1]
    else:
        interval_end = "0101"
        interval_start = dates[-1]

    return interval_start,interval_end


def get_hdf5_keys(start_year:int, start_index:int, end_year:int,
                  end_index:int, slice_length=6):
    if slice_length is None:
        slice_length=6
    if 12 % slice_length !=0:
        raise ValueError("12 is not dividable by time slice length {0}!".format(slice_length))

    dates = get_interval_dates(slice_length)

    keys=[]
    start_point = "{0}/".format(start_year)+dates[start_index]
    end_point = "{0}/".format(end_year)+dates[end_index]
    t1 = start_point
    yr = start_year
    k = start_index
    while t1 < end_point:
        t0 = t1
        if k==len(dates)-1:
            interval_end_date = dates[0]
            yr+=1
        else:
            interval_end_date = dates[k + 1]
        keys.append(t0+"-"+interval_end_date)
        t1 = "{0}/".format(yr) + interval_end_date
        k = (k+1)%len(dates)

    return keys,slice_length


def get_time_kwargs(key):
    start,end = key.split("-")
    start_yr = start[:4]
    start_mon = start[5:7]
    start_day = start[7:9]
    end_mon = end[:2]
    end_day = end[2:4]
    if end_mon=="01" and end_day=="01":
        end_yr = str(int(start_yr)+1)
    else:
        end_yr = start_yr

    start = "-".join([start_yr,start_mon,start_day])
    end = "-".join([end_yr,end_mon,end_day])
    lowerbound = "-".join([str(int(start_yr)-10),start_mon,start_day])
    upperbound = "-".join([str(int(end_yr)+1),end_mon,end_day])

    return {"start":start,"end":end,"lowerbound":lowerbound,"upperbound":upperbound}


def dataset2hdf5(targets, paras, start_year:int, start_index:int, end_year:int,
                  end_index:int, slice_length=None, stock_pool=None):
    hdf5_keys,slice_length = get_hdf5_keys(
        start_year=start_year,start_index=start_index,
        end_year=end_year,end_index=end_index,slice_length=slice_length)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    f_path = r"datasets/stock_d.hdf"
    f_path = add_suffix_to_file_names(f_path, current_date)
    length = {}
    columns= None
    for key in hdf5_keys:
        kwargs = get_time_kwargs(key)
        df_feature, df_not_in_X, cols_category, enc = ml_model.gen_data(
            targets=targets, stock_pool=stock_pool,**kwargs)

        X = ml_model.gen_X(df_feature, df_not_in_X.columns)
        # X = df_feature

        Y = pd.concat([ml_model.gen_y(df_not_in_X, **v) for k, v in paras]
                      ,axis=1)
        Y.columns = [k for k, _ in paras]
        Y.index = X.index
        Y["y_l"] = Y.apply(
            lambda r: r["y_l_rise"] if r["y_l_rise"] > -r["y_l_decline"] else
            r["y_l_decline"], axis=1)
        print(X.shape, Y.shape, Y.columns)

        # df_not_in_X[["qfq_avg", "f1mv_qfq_avg"]] = df_not_in_X[
        #     ["qfq_avg", "f1mv_qfq_avg"]].fillna(float("inf"))
        df_not_in_X["delist_date"] = df_not_in_X["delist_date"].fillna("")

        length[key]=len(X)
        if columns is None:
            columns = {"X":X.columns,"Y":Y.columns,"other":df_not_in_X.columns}
        X.to_hdf(f_path, key="X/" + key)
        Y.to_hdf(f_path, key="Y/" + key)
        df_not_in_X.to_hdf(f_path, key="other/" + key)


    store = pd.HDFStore(f_path)
    print(store.keys())
    for key in sorted(store.keys()):
        print(key,store[key].shape)

    d_info = {"slice_length":slice_length, "length":length,"columns":columns}
    f_path = add_suffix_to_file_names(r"datasets/stock_d_info",current_date)
    with open(f_path,"wb") as file:
        pickle.dump(d_info,file)


def read_hdf5(start,end=None,version=None):
    """

    :param start: Inclusive start date of desired time slice, formatted as
        "%Y-%m-%d".
    :param end: Exclusive dnd date of desired time slice, formatted as
        "%Y-%m-%d".
    :param version: the date of file, formatted same as above.
    :return:
    """
    start_yr = start[:4]
    start_mon = start[5:7]
    start_day = start[8:10]

    if end is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        end_yr, end_mon, end_day = now.split("-")
    else:
        end_time = datetime.datetime.strptime(end,
                                          "%Y-%m-%d")-datetime.timedelta(
            days=1)
        end = datetime.datetime.strftime(end_time,"%Y-%m-%d")  # Inclusive.
        end_yr = end[:4]
        end_mon = end[5:7]
        end_day = end[8:10]

    base_dir = r"datasets/"
    if version is None:
        # version is None, choose the latest info file.
        f_list = os.listdir(base_dir)
        info_file = sorted([f for f in f_list if f[:12]=="stock_d_info"])[-1]
        version = info_file[-10:]
    else:
        info_file = "stock_d_info_{0}".format(version)

    with open(os.path.join(base_dir,info_file),"rb") as f:
        d_info = pickle.load(f)

    print(d_info)

    slice_length = d_info["slice_length"]
    dates = get_interval_dates(slice_length)
    start_interval = get_interval_mon_day(mon=start_mon,day=start_day,
                                          slice_length=slice_length,
                                          dates=dates)
    end_interval = get_interval_mon_day(mon=end_mon,day=end_day,
                                        slice_length=slice_length,dates=dates)
    start_key = start_yr+"/"+"-".join(start_interval)
    end_key = end_yr +"/"+"-".join(end_interval)


    store = pd.HDFStore(os.path.join(base_dir,"stock_d_{0}.hdf".format(version)))
    keys = sorted([key[-14:] for key in store.keys() if key[-14:]>=start_key
            and key[-14:]<=end_key])
    X_list = []
    Y_list = []
    other_list = []
    for k in keys:
        print(k)
        print(store["X/"+k].shape)
        X_list.append(store["X/"+k])
        Y_list.append(store["Y/"+k])
        other_list.append(store["other/"+k])
    X = pd.concat(X_list)
    print(X.shape)
    Y = pd.concat(Y_list)
    other = pd.concat(other_list)
    date_index = X.index[(X.index>=start) & (X.index<end)].unique()
    return X.loc[date_index],Y.loc[date_index],other.loc[date_index]


def add_suffix_to_file_names(files, suffix:str):
    """
    Add date suffix to given file names.

    :param files: A dict of file names.
    :param suffix:
    :return: A dict of file names with date suffix.
    """
    if type(files)==str:
        f_name = files
        if '.' in f_name:
            idx = f_name.rindex(".")
        else:
            idx = len(f_name)
        return (f_name[:idx] + "_{0}" + f_name[idx:]).format(suffix)
    elif type(files)==dict:
        files = files.copy()
        for k in files.keys():
            f_name = files[k]
            if '.' in f_name:
                idx = f_name.rindex(".")
            else:
                idx = len(f_name)
            files[k] = (f_name[:idx]+"_{0}"+f_name[idx:]).format(suffix)
        return files
    else:
        raise ValueError("files:{} is not supported!".format(type(files)))


if __name__ == '__main__':
    # keys,_ = get_hdf5_keys(2016,0,2018,0)
    # print(keys)
    #
    # print(get_time_kwargs(keys[0]))
    #
    # targets = [{"period": 20, "fun": "max", "col": "high"},
    #            {"period": 20, "fun": "min", "col": "low"},
    #            {"period": 5, "fun": "max", "col": "high"},
    #            {"period": 5, "fun": "min", "col": "low"},
    #            # {"period": 20, "fun": "mean", "col": ""}
    #            ]
    # paras = [("y_l_rise",
    #           {"pred_period": 20, "is_high": True, "is_clf": False,
    #            "threshold": 0.2}),
    #          ("y_l_decline",
    #           {"pred_period": 20, "is_high": False, "is_clf": False,
    #            "threshold": 0.2}),
    #          ("y_s_rise",
    #           {"pred_period": 5, "is_high": True,"is_clf": False,
    #            "threshold": 0.1}),
    #          ("y_s_decline",
    #           {"pred_period": 5,"is_high": False,"is_clf": False,
    #            "threshold": 0.1}), ]
    #
    # dataset2hdf5(targets=targets,paras=paras,
    #              start_year=2016,start_index=0,end_year=2018,end_index=0)
    # read_hdf5(start="2016-01-01",end="2018-01-01")

    X,Y,other = read_hdf5(start="2016-07-01",end="2017-01-01")
    print(X.shape,Y.shape,other.shape)
    print(X.index.max(),X.index.min())
    print(Y.index.max(), Y.index.min())
    print(other.index.max(), other.index.min())