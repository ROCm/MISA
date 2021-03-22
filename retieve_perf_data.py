import re

def get_log_lines(log_file_name):
    with open(log_file_name, 'r') as f:
        log_lines = f.readlines()
        return log_lines

def get_runtime(log_file_name):
    txt_lines = get_log_lines(log_file_name)
    min_cost = []
    min_kernel = []
    sel_kernel = []
    sel_costs = []

    for each_line in txt_lines:
        res_str = re.search(r'(?<=fastest:).*', each_line)
        if res_str:
            res_str = re.search(r'(?<=tflops:)\d+\.?\d*', each_line)
        if res_str:
            driver_cost = float(res_str.group())
            print(f"driver_cost={driver_cost}")
            min_cost.append(driver_cost)
        res_str = re.search(r'kernel_name:', each_line)
        if res_str:
            min_kernel.append(each_line.split(":")[-1][:-1])
            print(each_line.split(":")[-1][:-1])
        res_str = re.search(r'selected kernel:', each_line)
        if res_str:
            sel_kernel.append(each_line.split(":")[-1][:-1])
            print(each_line.split(":")[-1][:-1])

        res_str = re.search(r'(?<=selected cost:)\d+\.?\d*', each_line)
        if res_str:
            sel_cost = float(res_str.group())
            print(f"driver_cost={sel_cost}")
            sel_costs.append(sel_cost)

    with open("./wrw_model.csv", "w") as f:
        for num_cost in min_cost:
            f.write(f"{num_cost}")
            f.write("\n")
        for sel_cost in sel_costs:
            f.write(f"{sel_cost}")
            f.write("\n")
        for kernel_name in min_kernel:
            f.write(f"{kernel_name}")
            f.write("\n")
        for s_kernel_name in sel_kernel:
            f.write(f"{s_kernel_name}")
            f.write("\n")

def get_kernel_name(log_file_name):
    txt_lines = get_log_lines(log_file_name)
    section_lines = []
    store_line = 0
    kernel_names = []
    for each_line in txt_lines:
        conv_line = re.match(r"(?!#)./out/conv_driver.exe conv", each_line)
        if store_line:
            kernel_name = re.match(r"kernel:", each_line)
            if kernel_name:
                name = each_line.split(':')[-1]
                print(f"\t\"{name[:-1]}\",")
                kernel_names.append(name)
        if conv_line:
            store_line = 1
        conv_end_line = re.match(r"min cost:", each_line)
        if conv_end_line:
            break
    
    return kernel_names


if __name__ == '__main__':
    #names = get_kernel_name("./wrw_model.log")
    #print(len(names))
    get_runtime("./fwd_fp32_nhwc_models.log")
