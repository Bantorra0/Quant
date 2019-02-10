import numpy as np
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


def save_dataset_in_hdf5(targets, paras, start_year:int, start_index:int, end_year:int,
                         end_index:int, slice_length=None, stock_pool=None,
                         version=None, base_dir=r"datasets",f_info_name="stock_d_info"):
    hdf5_keys,slice_length = get_hdf5_keys(
        start_year=start_year,start_index=start_index,
        end_year=end_year,end_index=end_index,slice_length=slice_length)

    if version is None:
        version = datetime.datetime.now().strftime("%Y-%m-%d")
    f_hdf_name = "stock_d.hdf"
    f_hdf_name = _add_suffix_to_file_names(f_hdf_name, version)
    f_hdf_path = os.path.join(base_dir,f_hdf_name)
    if f_hdf_name in os.listdir(base_dir):
        store = pd.HDFStore(f_hdf_path)
    else:
        store = None
    # print("store",store)

    if f_info_name is None:
        f_info_name = "stock_d_info"
    f_info_name = _add_suffix_to_file_names(f_info_name, version)
    f_info_path = os.path.join(base_dir,f_info_name)
    if f_info_name in os.listdir(base_dir):
        with open(f_info_path,"rb") as f_info:
            d_info = pickle.load(f_info)
    else:
        d_info = {"slice_length": slice_length, "length": {},
                  "columns": None, "categorical_features": None,
                  "encoder": None}
    for key in hdf5_keys:
        print("key:",key)
        skip = False

        if store is not None \
                and "/X/"+key in store.keys() \
                and "/Y/"+key in store.keys() \
                and "/other/"+key in store.keys():
                # X,Y,df_other = store["X/"+key],store["Y/"+key],store["other/"+key]
            skip = True

        if not skip:
            kwargs = get_time_kwargs(key)
            X, df_other, cols_category, enc = ml_model.gen_data(
                targets=targets, stock_pool=stock_pool, **kwargs)
            # X = ml_model.gen_X(df_feature, df_other.columns)

            Y = pd.concat([ml_model.gen_y(df_other, **v) for k, v in paras],
                          axis=1)
            Y.columns = [k for k, _ in paras]
            Y.index = X.index
            Y["y_l"] = Y.apply(lambda r: r["y_l_rise"] if r["y_l_rise"] > -r[
                "y_l_decline"] else r["y_l_decline"], axis=1)
            print(X.shape, Y.shape, Y.columns)

            # df_other[["qfq_avg", "f1mv_qfq_avg"]] = df_other[
            #     ["qfq_avg", "f1mv_qfq_avg"]].fillna(float("inf"))
            df_other["delist_date"] = df_other["delist_date"].fillna("")

            d_info["length"][key] = len(X)
            if d_info["columns"] is None:
                d_info["columns"] = {"X": X.columns, "Y": Y.columns,
                                     "other": df_other.columns}
            if d_info["categorical_features"] is None:
                d_info["categorical_features"] = cols_category
            if d_info["encoder"] is None:
                d_info["encoder"] = enc

            X.to_hdf(f_hdf_path, key="X/" + key)
            Y.to_hdf(f_hdf_path, key="Y/" + key)
            df_other.to_hdf(f_hdf_path, key="other/" + key)
            del X,Y,df_other
        else:
            print("skip:", key)
            # d_info["length"][key] = len(store["/X/"+key])

    if store is None:
        store = pd.HDFStore(f_hdf_path)
    print(store.keys())
    for key in sorted(store.keys()):
        print(key,store[key].shape)
    store.close()

    print(d_info)
    with open(f_info_path,"wb") as f_info:
        pickle.dump(d_info,f_info)


def fill_paras(**kwargs):
    paras = kwargs.copy()
    if "base_dir" not in kwargs or kwargs["base_dir"] is None:
        paras["base_dir"] = "datasets"

    if "f_info_name" not in kwargs or kwargs["f_info_name"] is None:
        paras["f_info_name"] = "stock_d_info"

    if "version" not in kwargs or kwargs["version"] is None:
        paras["version"] = get_latest_version(paras["base_dir"],
                                              paras["f_info_name"])

    return paras


def get_f_info_path(version=None, base_dir=None,
                      f_info_name=None):
    paras = fill_paras(version=version,base_dir=base_dir,
                      f_info_name=f_info_name)
    f_info_name = paras["f_info_name"]
    version = paras["version"]
    base_dir = paras["base_dir"]

    f_info_name = _add_suffix_to_file_names(f_info_name, version)
    f_info_path = os.path.join(base_dir, f_info_name)
    return f_info_path,f_info_name


def load_dataset_info(version=None, base_dir=None,
                      f_info_name=None):
    paras = fill_paras(version=version, base_dir=base_dir,
                       f_info_name=f_info_name)
    f_info_name = paras["f_info_name"]
    version = paras["version"]
    base_dir = paras["base_dir"]

    f_info_path,f_info_name = get_f_info_path(version,base_dir,f_info_name)
    if f_info_name not in os.listdir(base_dir):
        raise ValueError("{0} not is directory {1}".format(f_info_name,base_dir))
    with open(f_info_path,"rb") as f_info:
        d_info = pickle.load(f_info)
    return d_info


def save_dataset_info(d_info,version=None, base_dir=None,
                      f_info_name=None):
    f_info_path,_ = get_f_info_path(version,base_dir,f_info_name)
    with open(f_info_path, "wb") as f_info:
        pickle.dump(d_info, f_info)


def save_shuffle_info(n=4,version=None, base_dir=None,
                      f_info_name=None):
    d_info = load_dataset_info(version,base_dir,f_info_name)

    shuffles = {}
    for k,v in d_info["length"].items():
        a = np.arange(v)
        for i in range(n):
            np.random.shuffle(a)
        shuffles[k] = a

    d_info["shuffle"] = shuffles
    print(d_info)
    save_dataset_info(d_info,version,base_dir,f_info_name)


def read_hdf5(start,end=None,version=None,base_dir = None,
              f_info_name=None, subsample=None):
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
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        end_yr, end_mon, end_day = end.split("-")
    else:
        end_time = datetime.datetime.strptime(end,"%Y-%m-%d")\
                   -datetime.timedelta(days=1)
        end = datetime.datetime.strftime(end_time,"%Y-%m-%d")  # Inclusive.
        end_yr = end[:4]
        end_mon = end[5:7]
        end_day = end[8:10]
    print(start,end)

    paras = fill_paras(version=version, base_dir=base_dir,
                       f_info_name=f_info_name)
    f_info_name = paras["f_info_name"]
    version = paras["version"]
    base_dir = paras["base_dir"]

    d_info = load_dataset_info(version,base_dir,f_info_name)
    # print(d_info)

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
            and key[-14:]<=end_key and key[1]=="X"])
    print("Time slice keys in hdf5:",",".join(keys))

    X_list = []
    Y_list = []
    other_list = []
    for key in keys:
        print("\nCurrent key:",key)
        print("Current slice size(length):",store["X/"+key].shape[0])
        if subsample is None:
            X, Y,df_other=store["X/"+key],store["Y/"+key],store["other/"+key]
        else:
            k,ith = tuple(map(int,subsample.split("-")))
            shuffle = d_info["shuffle"][key]
            length = d_info["length"][key]
            n = length//k
            if k-1==ith:
                idx = shuffle[ith * n:]
            else:
                idx = shuffle[ith*n:(ith+1)*n]
            X = store["X/"+key].iloc[idx]
            Y = store["Y/"+key].iloc[idx]
            df_other = store["other/"+key].iloc[idx]
            print("Current subsample size(length):",len(idx))
        X_list.append(X)
        Y_list.append(Y)
        other_list.append(df_other)
    store.close()
    X = pd.concat(X_list)
    print("\nTotal concatenating size:",X.shape[0])
    Y = pd.concat(Y_list)
    other = pd.concat(other_list)
    date_index = X.index[(X.index>=start) & (X.index<end)].unique()
    print("Result dataset size:",len(Y.loc[date_index]))
    return X.loc[date_index],Y.loc[date_index],other.loc[date_index]


def _add_suffix_to_file_names(files, suffix:str):
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
            files[k] = _add_suffix_to_file_names(files[k], suffix)
        return files
    else:
        raise ValueError("files:{} is not supported!".format(type(files)))


def get_latest_version(base_dir=r"datasets",
                      f_info_name="stock_d_info"):
    if base_dir is None:
        base_dir = r"datasets"
    if f_info_name is None:
        f_info_name = "stock_d_info"

    f_list = os.listdir(base_dir)
    info_file = sorted([f for f in f_list if f[:12] == f_info_name])[-1]
    version = info_file[-10:]
    return version


if __name__ == '__main__':
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
    # save_dataset_in_hdf5(targets=targets, paras=paras,
    #                      start_year=2013, start_index=0, end_year=2019,
    #                      end_index=0,
    #                      version="2019-02-06")


    # X,Y,other = read_hdf5(start="2016-07-01",end="2017-01-01")
    d_info = load_dataset_info()
    # for k,v in sorted(d_info["shuffle"].items()):
    #     print(k,v)
    for k,v in sorted(d_info["length"].items()):
        print(k,v)

    X, Y, df_other = read_hdf5(start="2016-07-01", end="2017-01-01",
                               subsample="10-5")
    print(X.shape, Y.shape, df_other.shape)
    print(X.index.max(),X.index.min())
    print(Y.index.max(), Y.index.min())
    print(df_other.index.max(), df_other.index.min())

    # save_shuffle_info()


