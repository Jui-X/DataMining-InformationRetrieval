import xlsxwriter
import sys, getopt


def load_data(input):
    transactions = []
    count = 0
    with open(input) as f:
        content = f.readlines()
        for lines in content:
            line = lines.strip('\n')
            nums = line.split(' ')
            num = frozenset(nums)
            transactions.append(num)

    f.close()

    return transactions


# 扫描并对数据库进行support_count
def support_count_C1(transactions):
    dic = dict.fromkeys(range(120), 0)

    for items in transactions:
        for item in items:
            item = int(item)
            dic[item] += 1

    keys = []
    for key in dic.keys():
        if dic[key] == 0:
            keys.append(key)

    for key in keys:
        dic.pop(key)

    row = 1
    col = 0
    # write_into_excel(dic, row, col)

    return dic


# 扫描每一个候选集ck并统计计数
def support_count(transactions, Ck):
    Ck_support = {}
    for transaction in transactions:
        for item in Ck:
            # 如果候选项是事务的子集
            if item.issubset(transaction):
                if item in Ck_support:
                    Ck_support[item] += 1
                else:
                    Ck_support[item] = 1

    return Ck_support


# 将每个项的support_count写入Excel表格
def write_into_excel(dic, row, col):
    workbook = xlsxwriter.Workbook('./statistic_data.xlsx')
    worksheet = workbook.add_worksheet()
    bold_format = workbook.add_format({'bold': True})

    # 表格从0列到1列，有100行
    worksheet.set_column(1, 100)
    # 两列的标题
    worksheet.write('A1', 'support_count', bold_format)

    # 从第一行开始，写入0 1两列
    for key in dic.keys():
        worksheet.write_number(row, col, key)
        worksheet.write_number(row, col + 1, dic[key])
        row += 1

    workbook.close()


# 将频繁集写入Excel表格
def Apriori_output(Lk, output):
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    bold_format = workbook.add_format({'bold': True})

    # 表格从0列到1列，有100行
    worksheet.set_column(1, 100)
    # 两列的标题
    worksheet.write('A1', '频繁1项集', bold_format)
    worksheet.write('B1', '频繁2项集', bold_format)
    worksheet.write('C1', '频繁3项集', bold_format)
    worksheet.write('D1', '频繁4项集', bold_format)
    worksheet.write('E1', '频繁5项集', bold_format)
    worksheet.write('F1', '频繁6项集', bold_format)
    worksheet.write('G1', '频繁7项集', bold_format)
    worksheet.write('H1', '频繁8项集', bold_format)
    worksheet.write('I1', '频繁9项集', bold_format)
    worksheet.write('J1', '频繁10项集', bold_format)
    worksheet.write('K1', '频繁11项集', bold_format)
    worksheet.write('L1', '频繁12项集', bold_format)
    worksheet.write('M1', '频繁13项集', bold_format)

    # 从第一行开始，写入0 1两列
    row = 1
    col = 0
    for L in Lk:
        row = 1
        for frequent_item in L:
            worksheet.write_string(row, col, str(frequent_item))
            row += 1
        col += 1

    workbook.close()


# 将候选1项集中的support_count与min_sup比较，生成频繁1项集
def Apriori_L1(C1_support, data_count, min_sup):
    '''
    :param ck: 候选项集
    :param min_sup: 最小支持度阈值
    :return: 返回频繁1项集
    '''

    L1 = set()
    for item in C1_support.keys():
        if (float(C1_support[item]) / data_count) > float(min_sup):
            # 生成频繁1项集时用frozenset数据结构来存放1频繁项
            frequent_item = frozenset([str(item)])
            L1.add(frequent_item)

    return L1


# 将候选k项集中的support_count与min_sup比较，生成频繁k项集
def Apriori(Ck_support, data_count, min_sup):
    '''
    用于将精简后的ck进行统计得到ck_support
    将ck_support与min_sup进行比较筛选出频繁集
    :param ck_support: 经统计后ck中候选项的计数
    :param min_sup: 最小支持度阈值
    :return: 返回频繁项集
    '''
    Lk = set()
    for item in Ck_support.keys():
        if (float(Ck_support[item]) / data_count) > float(min_sup):
            Lk.add(item)

    return Lk


# 生成ck候选项集
def Apriori_gen(Lk_sub_1, k):
    '''
    连接步：将Lk-1与自身连接，当两个元素前k-2个项相同时，才进行来连接
    目的是减少遍历次数，确保不产生重复项
    :param lk_sub_1: 表示频繁k-1项集Lk-1，用于构造候选集Ck
    :param k: 当前候选集的项数
    :return: 生成ck候选项集
    '''
    Ck = set()
    l1 = []
    l2 = []
    Lk_len = len(Lk_sub_1)
    Lk = list(Lk_sub_1)
    for i in range(Lk_len):
        for j in range(i+1, Lk_len):
            l1 = list(Lk[i])
            l2 = list(Lk[j])
            l1.sort()
            l2.sort()

            # 如果l1和l2的前k-2项相同，则将两项连接生成一个新的候选项
            if l1[:k-2] == l2[:k-2]:
                Ck_item = Lk[i] | Lk[j]
                if is_Apriori(Ck_item, Lk_sub_1):
                    Ck.add(Ck_item)

    # 利用先验性质除去候选项集ck中的非频繁项集
    for C in Ck:
        if not is_Apriori(C, Lk_sub_1):
            Ck.remove(C)

    return Ck


# 剪枝，如果候选项集的子集不在频繁k-1项集Lk-1中
def is_Apriori(Ck, Lk_sub_1):
    '''
    用于剪枝步：利用先验性质删除ck中一定不是频繁集的候选项
    :param ck: 当前候选集ck
    :param lk_sub_1: 频繁k-1项集
    :return: 当前连接得到的候选项是否频繁
    '''
    for item in Ck:
        sub_item = Ck - frozenset([item])
        if sub_item not in Lk_sub_1:
            return False
    return True


# def generate_Rule(Lk, C_support, min_sup, min_conf):
#     rule = []
#     sub_set = set()
#     for L in Lk:
#         for frequent_item in L:
#             sub_set.add(frequent_item)
#             for sub in sub_set:
#                 if sub.issubset(frequent_item):
#                     conf = C_support[frequent_item] / C_support[sub_set]
#                     if conf >= min_conf:
#                         rule.append([sub_set, frequent_item - sub_set])
#
#     return rule


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

    # count = 0
    # for l in L:
    #     count += len(l)
    # print(count)
    Apriori_output(L, output_filename)

