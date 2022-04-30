import pandas as pd


def csv_to_latex(inp_path, out_path):
    latex_str = ""
    df = pd.read_csv(inp_path)
    for _, row_series in df.iterrows():
        indexandrow = row_series.tolist()
        index = indexandrow[0]
        row = indexandrow[1:]
        row_str = ""
        row_str += index + " & "
        for i in range(len(row) // 2):
            truncated_mean = str("{:0.2f}".format(abs(row[2 * i])))
            truncated_confidence = str("{:0.2f}".format(row[2 * i + 1]))
            row_str += truncated_mean + " $\\pm$ " + truncated_confidence
            if 2 * i + 1 != len(row) - 1:
                row_str += " & "

        row_str += " \\\ \\hline "
        latex_str += row_str

    with open(out_path, "w") as f:
        f.write(latex_str)


if __name__ == "__main__":
    csv_to_latex("../evaluation/two_thinning/data/comparison.csv", "../evaluation/two_thinning/data/comparison.tex")
    csv_to_latex("../evaluation/k_thinning/data/comparison.csv", "../evaluation/k_thinning/data/comparison.tex")
    csv_to_latex("../evaluation/graphical_two_choice/data/comparison.csv", "../evaluation/graphical_two_choice/data/comparison.tex")
