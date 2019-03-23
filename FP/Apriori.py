import xlsxwriter


def load_data():
    transactions = []
    with open("homework1-1.txt") as f:
        content = f.readlines()
        for lines in content:
            line = lines.strip('\n')
            nums = line.split(' ')
            num = frozenset(nums)
            transactions.append(num)

    f.close()

    return transactions


# 扫描并对数据库进行support_count
def support_count_c1(transactions):
    dic = dict.fromkeys(range(120), 0)

    for items in transactions:
        for item in items:
            item = int(item)
            dic[item] = dic[item] + 1

    keys = []
    for key in dic.keys():
        if dic[key] == 0:
            keys.append(key)

    for key in keys:
        dic.pop(key)

    # write_into_excel(dic)

    return dic


def support_count(transactions, ck):
    ck_support = {}
    for transaction in transactions:
        for item in ck:
            if item.issubset(transaction):
                if item in ck_support:
                    ck_support[item] += 1
                else:
                    ck_support[item] = 1

    return ck_support


# 将每个项的support_count写入Excel表格
def write_into_excel(dic):
    workbook = xlsxwriter.Workbook('./statistic_data.xlsx')
    worksheet = workbook.add_worksheet()
    bold_format = workbook.add_format({'bold': True})

    worksheet.set_column(1, 100, 15)
    worksheet.write('A1', 'support_count', bold_format)

    row = 1
    col = 0
    for key in dic.keys():
        worksheet.write_number(row, col, key)
        worksheet.write_number(row, col + 1, dic[key])
        row += 1

    workbook.close()


def Apriori_L1(ck, min_sup):
    '''
    :param ck: 候选项集
    :param min_sup: 最小支持度阈值
    :return: 返回频繁1项集
    '''

    l1 = set()
    for item in ck.keys():
        if ck[item] > min_sup:
            frequent_item = frozenset(str(item))
            l1.add(frequent_item)

    return l1


def Apriori(ck_support, min_sup):
    '''
    用于将精简后的ck进行统计得到ck_support
    将ck_support与min_sup进行比较筛选出频繁集
    :param ck_support: 经统计后ck中候选项的计数
    :param min_sup: 最小支持度阈值
    :return: 返回频繁项集
    '''
    lk = set()
    for item in ck_support.keys():
        if ck_support[item] > min_sup:
            frequent_item = frozenset(str(item))
            lk.add(frequent_item)

    return lk


def Apriori_gen(lk_sub_1, k):
    '''
    连接步：将Lk-1与自身连接，当两个元素前k-2个项相同时，才进行来连接
    目的是减少遍历次数，确保不产生重复项
    :param lk_sub_1: 表示频繁k-1项集Lk-1，用于构造候选集Ck
    :param k: 当前候选集的项数
    :return: 生成ck候选项集
    '''
    ck = set()
    l1 = []
    l2 = []
    lk_len = len(lk_sub_1)
    lk = list(lk_sub_1)
    for i in range(lk_len):
        for j in range(i+1, lk_len):
            l1 = list(lk[i])
            l2 = list(lk[j])
            l1.sort()
            l2.sort()

            # 如果l1和l2的前k-2项相同，则将
            if l1[:k-2] == l2[:k-2]:
                ck_item = lk[i] | lk[j]
                if is_Apriori(ck_item, lk_sub_1):
                    ck.add(frozenset(ck_item))

    for c in ck:
        if not is_Apriori(c, lk_sub_1):
            ck.remove(c)

    return ck


def is_Apriori(ck, lk_sub_1):
    '''
    用于剪枝步：利用先验性质删除ck中一定不是频繁集的候选项
    :param ck: 当前候选集ck
    :param lk_sub_1: 频繁k-1项集
    :return: 当前连接得到的候选项是否频繁
    '''
    for item in ck:
        sub_item = ck - frozenset([item])
        if sub_item not in lk_sub_1:
            return False

    return True


if __name__ == '__main__':
    min_sup = 1000
    L = [[]]
    c = [[], []]
    c_support = [[]]

    data = load_data()
    c_support.append(support_count_c1(data))
    L.append(Apriori_L1(c_support[1], min_sup))

    # print(L)
    k = 2

    while L[k-1]:
        c.append(Apriori_gen(L[k - 1], k))
        c_support.append(support_count(data, c[k]))
        L.append(Apriori(c_support[k], min_sup))
        k += 1

    print(L)
    print(c)
    print(c_support)

