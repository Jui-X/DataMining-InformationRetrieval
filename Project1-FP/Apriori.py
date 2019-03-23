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
def support_count(transactions):
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

    Lk = []
    for item in ck.keys():
        if ck[item] > min_sup:
            Lk.append(item)

    return Lk


def Apriori(ck_support, min_sup):
    '''
    用于将精简后的ck进行统计得到
    :param ck_support:
    :param min_sup:
    :return: 返回频繁项集
    '''
    lk = []
    for item in ck_support.keys():
        if ck_support[item] > min_sup:
            lk.append(item)

    return lk


def Apriori_gen(lk_sub_1, k):
    '''
    连接步：将Lk-1与自身连接，当两个元素前k-2个项相同时，才进行来连接
    目的是减少遍历次数，确保不产生重复
    :param lk_sub_1: 表示频繁k-1项集Lk-1，用于构造候选集Ck
    :param k: 当前候选集的项数
    :return: 生成ck候选项集
    '''
    ck = []
    l1 = []
    l2 = []
    lk_len = len(lk_sub_1)
    for i in range(lk_len):
        for j in range(i+1, lk_len):
            l1 = lk_sub_1[i][:k-2]
            l2 = lk_sub_1[j][:k-2]

            # 如果l1和l2的前k-2项相同，则将
            if l1 == l2:
                ck_item = lk_sub_1[i] | lk_sub_1[j]
                if is_Apriori(ck_item, lk_sub_1):
                    ck.append(ck_item)

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
    c_support = [[]]

    transaction = load_data()
    c_support.append(support_count(transaction))
    L.append(Apriori_L1(c_support[1], min_sup))

    print(L)
    k = 2

    c_support.append(Apriori())

    # while L[k-1]:
    #     Ck = Apriori_gen(L[k-1], k)
    #     k += 1

    #     L[k] = L[k-1]
