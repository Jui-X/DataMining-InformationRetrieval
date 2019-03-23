'''
# Filename: MyApriori.py
# author: 薄层
# Date: 2018/5/20 0:41
'''


def load_data_set():
    data_set = [['e1', 'e2', 'e5'],
                ['e2', 'e4'],
                ['e2', 'e3'],
                ['e1', 'e2', 'e4'],
                ['e1', 'e3'],
                ['e2', 'e3'],
                ['e1', 'e3'],
                ['e1', 'e2', 'e3', 'e5'],
                ['e1', 'e2', 'e3']]
    return data_set


def Create_C1(data_set):
    '''
	参数：数据库事务集
	'''
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            # 为生成频繁项目集时扫描数据库时以提供issubset()功能
            C1.add(item_set)
    return C1


def is_apriori(Ck_item, Lk_sub_1):
    '''
	参数：候选频繁k项集，频繁k-1项集
	'''
    for item in Ck_item:
        sub_item = Ck_item - frozenset([item])
        if sub_item not in Lk_sub_1:
            return False
    return True


def Create_Ck(Lk_sub_1, k):
    '''
	# 参数：频繁k-1项集，当前要生成的候选频繁几项集
	'''
    Ck = set()
    len_Lk_sub_1 = len(Lk_sub_1)
    list_Lk_sub_1 = list(Lk_sub_1)
    for i in range(len_Lk_sub_1):  # i: [0, len_Lk_sub_1)
        for j in range(i + 1, len_Lk_sub_1):  # j: [i+1, len_Lk_sub_1)
            l1 = list(list_Lk_sub_1[i])
            l2 = list(list_Lk_sub_1[j])
            l1.sort()
            l2.sort()
            # 判断l1的前k-1-1个元素与l2的前k-1-1个元素对应位是否全部相同
            # list[s:t]：截取s到t范围的元素生成一个新list
            if l1[0:k - 2] == l2[0:k - 2]:
                Ck_item = list_Lk_sub_1[i] | list_Lk_sub_1[j]
                if is_apriori(Ck_item, Lk_sub_1):
                    Ck.add(Ck_item)
    return Ck


def Generate_Lk_By_Ck(data_set, Ck, min_support, support_data):
    '''
	参数：数据库事务集，候选频繁k项集，最小支持度，项目集-支持度dic
	'''
    Lk = set()
    # 通过dic记录候选频繁k项集的事务支持个数
    item_count = {}
    for t in data_set:
        for Ck_item in Ck:
            if Ck_item.issubset(t):
                if Ck_item not in item_count:
                    item_count[Ck_item] = 1
                else:
                    item_count[Ck_item] += 1
    data_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / data_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / data_num
    return Lk


def Generate_L(data_set, max_k, min_support):
    '''
	参数：数据库事务集，求的最高项目集为k项，最小支持度
	'''
    # 创建一个频繁项目集为key，其支持度为value的dic
    support_data = {}
    C1 = Create_C1(data_set)
    L1 = Generate_Lk_By_Ck(data_set, C1, min_support, support_data)
    Lk_sub_1 = L1.copy()  # 对L1进行浅copy
    L = []
    L.append(Lk_sub_1)  # 末尾添加指定元素
    for k in range(2, max_k + 1):
        Ck = Create_Ck(Lk_sub_1, k)
        Lk = Generate_Lk_By_Ck(data_set, Ck, min_support, support_data)
        Lk_sub_1 = Lk.copy()
        L.append(Lk_sub_1)
    return L, support_data


def Generate_Rule(L, support_data, min_confidence):
    '''
	参数：所有的频繁项目集，项目集-支持度dic，最小置信度
	'''
    rule_list = []
    sub_set_list = []
    for i in range(len(L)):
        for frequent_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(frequent_set):
                    conf = support_data[frequent_set] / support_data[sub_set]
                    # 将rule声明为tuple
                    rule = (sub_set, frequent_set - sub_set, conf)
                    if conf >= min_confidence and rule not in rule_list:
                        rule_list.append(rule)
            sub_set_list.append(frequent_set)
    return rule_list


if __name__ == "__main__":
    data_set = load_data_set()
    '''
	print("Test")
	# 数据库事务打印
	for t in data_set:
		print(t)
	'''
    '''
	print("Test")
	# 候选频繁1项集打印
	C1 = Create_C1(data_set)
	for item in C1:
		print(item)
	'''
    '''
	# 频繁1项集打印
	print("Test")
	L = Generate_L(data_set, 1, 0.2)
	for item in L:
		print(item)
	'''
    '''
	# 频繁k项集打印
	print("Test")
	L, support_data = Generate_L(data_set, 2, 0.2)
	for item in L:
		print(item)
	'''
    '''
	# 关联规则测试
	print("Test")
	L, support_data = Generate_L(data_set, 3, 0.2)
	rule_list = Generate_Rule(L, support_data, 0.7)
	for item in support_data:
		print(item, ": ", support_data[item])
	print("-----------------------")
	for item in rule_list:
		print(item[0], "=>", item[1], "'s conf:", item[2])
	'''

    L, support_data = Generate_L(data_set, 3, 0.2)
    rule_list = Generate_Rule(L, support_data, 0.7)
    for Lk in L:
        # print("=" * 55)
        print("frequent " + str(len(list(Lk)[0])) + "-itemsets\t\tsupport")
        # print("=" * 55)
        for frequent_set in Lk:
            print(frequent_set, support_data[frequent_set])
    print()
    # print("Rules")
    # for item in rule_list:
    #     print(item[0], "=>", item[1], "'s conf: ", item[2])
