import re
import pandas as pd


def WSL_position_string_proc(date,sentinel = ",,,"):
    """
    卫斯理公众号仓位统计结果的字符串预处理，结果用于excel中计算散户平均仓位。
    :param sentinel:
    :return:
    """

    print("Paste string below:")
    s = '\n'.join(iter(input, sentinel))
    # print(re.sub("票[0-9]+%", "", s).replace("\n\n", ",,,").replace("\n", "\t").replace(",,,", "\n"))
    print(s.replace("票","\t").replace("\n\n", ",,,").replace("\n", "\t").replace(",,,", "\n"))

    # s2 = re.sub("票[0-9]+%", "", s).replace("\n\n", ",,,").replace("\n", "\t").replace(",,,", "\n")
    # df = pd.DataFrame([line.split("\t") for line in s2.split("\n")],columns=["position range","count"])
    # positions = [0]+list(range(5,100,10))+[100,100,None]
    # df["position"] = positions
    # tot_cnt = df["count"].sum()
    # tot_valid_cnt = df["count"][:-1].sum()
    # avg_position = (df["count"]*df["position"]).sum()/tot_valid_cnt


if __name__ == '__main__':
    WSL_position_string_proc("20191014")


