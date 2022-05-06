import pandas as pd

NAME_DICT = {
    "dqn": "\\DQN",
    "always_accept": "\\AlwaysAccept",
    "random": "\\Random",
    "threshold": "\\Threshold",
    "dp": "\\DP",
    "mean_thinning": "\\MeanThinning",
    "local_reward_optimiser": "\\LocalRewardOptimiser",
    "quantile": "\\Quantile",
    "greedy": "\\Greedy",
}

def csv_to_latex(inp_path, out_path, top_highlight=1, name_dict=NAME_DICT):
    latex_str = ""
    df = pd.read_csv(inp_path)
    for _, row_series in df.iterrows():
        indexandrow = row_series.tolist()
        index = indexandrow[0]
        row = indexandrow[1:]
        row_str = ""
        row_str += name_dict[index] + " & "
        for i in range(len(row) // 2):
            avg = row[2 * i]
            if avg != -1:
                contestants = df.iloc[:, 2 * i + 1].tolist()
                conf = row[2 * i + 1]
                better_cnt = len([x for x in contestants if x != -1 and x < avg])
                need_bf = better_cnt < top_highlight and avg != -1
                start_bf = "\\textbf{" if need_bf else ""
                end_bf = "}" if need_bf else ""
                truncated_mean_str = str("{:0.2f}".format(avg))
                truncated_conf_str = str("{:0.2f}".format(conf))
                pm_str = " $\\pm$ "
                row_str += start_bf + truncated_mean_str + pm_str + truncated_conf_str + end_bf
            else:
                row_str += "TLE"
            if 2 * i + 1 != len(row) - 1:
                row_str += " & "

        row_str += " \\\ \\hline "
        latex_str += row_str

    with open(out_path, "w") as f:
        f.write(latex_str)


if __name__ == "__main__":
    # csv_to_latex("../evaluation/two_thinning/data/comparison.csv", "../evaluation/two_thinning/data/comparison.tex")
    # csv_to_latex("../evaluation/k_thinning/data/comparison.csv", "../evaluation/k_thinning/data/comparison.tex")
    csv_to_latex("../evaluation/graphical_two_choice/data/comparison.csv", "../evaluation/graphical_two_choice/data/comparison.tex")
