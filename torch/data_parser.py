import re


def parse(filename, mp=1):
    output = {}
    txt = open(filename, "r").read()
    # MLP
    pattern = r"mp {} \| mean time\s*\d+\.\d+".format(mp)
    mlp_time_str = re.findall(pattern, txt)[0]
    mlp_time = float(re.findall(r"\d+\.\d+", mlp_time_str)[0])
    print(mlp_time)
    output["MLP"] = mlp_time
    keywords = ["AttnLayer", "RMS", "Rotary"]
    for k in keywords:
        pattern = r"{}.*time\s*\d+\.\d+".format(k)
        time_str = re.findall(pattern, txt)[0]
        time = float(re.findall(r"\d+\.\d+", time_str)[0])
        print(time)
        output[k] = time
    return output


if __name__ == "__main__":
    print(parse("logs/2024-05-04_17-47-35.txt"))
