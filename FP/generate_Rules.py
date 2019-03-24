from Apriori import *
# from FP.Apriori import *
import xlsxwriter
import sys, getopt


# 通过频繁集和支持度计数生成强关联规则
def generate_Rule(Lk, C_support, min_sup, min_conf):
    rule_list = []
    sub_set = set()
    rule = []
    for L in Lk:
        for frequent_item in L:
            # 将频繁集中的项遍历
            for sub in sub_set:
                # 如果该项是频繁集的子集，则计算confidence
                if sub.issubset(frequent_item):
                    i = len(frequent_item)
                    j = len(sub)
                    for s in sub:
                        if int(s) in C_support[j].keys():
                            # 用 频繁项的 support_count / 子集的support_count
                            conf = C_support[i][frequent_item] / C_support[j][int(s)]
                            rule = str(sub_set) + "  ==>  " + str(frequent_item - sub_set)
                            # 比较 confidence 与 min_confidence
                            if conf >= float(min_conf) and rule not in rule_list:
                                rule_list.append(rule)
            # 将频繁集加进去
            sub_set.add(frequent_item)

    return rule_list


# 将频繁集写入Excel表格
def association_rule_output(rules, output):
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    bold_format = workbook.add_format({'bold': True})

    # 两列的标题
    worksheet.write('A1', '关联规则', bold_format)

    # 从第一行开始，写入0 1两列
    row = 1
    col = 0
    for rule in rules:
        worksheet.write_string(row, col, str(rule))
        row += 1

    workbook.close()


def main(argv):
    try:
        args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('command args error!')
        sys.exit(2)

    return args


if __name__ == '__main__':
    args = main(sys.argv[1:])

    min_sup = args[1][0]
    min_conf = args[1][1]
    input_filename = args[1][2]
    output_filename = args[1][3]

    L = [[]]
    C = [[], []]
    C_support = [[]]

    data_set = load_data(input_filename)
    data_count = float(len(data_set))
    C_support.append(support_count_C1(data_set))
    L.append(Apriori_L1(C_support[1], data_count, min_sup))

    # print(L[1])
    k = 1

    while L[k]:
        k += 1
        C.append(Apriori_gen(L[k - 1], k))
        C_support.append(support_count(data_set, C[k]))
        L.append(Apriori(C_support[k], data_count, min_sup))

    rule = []
    rule = generate_Rule(L, C_support, min_sup=min_sup, min_conf=min_conf)

    association_rule_output(rule, output_filename)