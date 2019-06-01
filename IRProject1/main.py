import os
import re
import jieba

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

path = "data/Project_Data"
files = os.listdir(path)


def load_data():
    data = []
    fileName = []
    for file in files:
        file = path + "/" + file
        with open(file, "r", encoding="utf8") as f:
            lines = []
            for line in f.readlines():
                if line != "--- #start#页 ---\n" and line != "--- #end#页 ---\n":
                    lines.append(line)
            lines.append(os.path.splitext(file)[0].split("/")[-1])
            data.append(lines)

    # with open("data/Project_Data/0b098ab4-75a8-4f39-90fe-db1447e96cf9_TERMS.txt", "r", encoding="utf8") as fp:
    #     datas = fp.readlines()
    #     for line in datas:
    #         if line != "--- #start#页 ---\n" and line != "--- #end#页 ---\n":
    #             data.append(line)

    return data


def product_match(dataset):
    product_category = []
    product_name = []
    product_payment = []
    product_term = []
    file_name = []

    for data in dataset:

        file_name.append(data[-1])

        name_re = u"[\u4e00-\u9fa5]{5,13}\u75be\u75c5\u4fdd\u9669(\u6761\u6b3e)*"
        r = re.compile(name_re)
        flag = False
        for line in data:

            l = r.search(line)
            if l:
                seg_list = jieba.cut(l.group(0), cut_all=False, HMM=True)
                seg_list = list(seg_list)
                name = []
                if len(seg_list[0]) != 1:
                    for seg in seg_list:
                        name.append(seg)
                else:
                    for i in range(1, len(seg_list)):
                        name.append(seg_list[i])
                name = "".join(name)
                product_name.append(name)

                basic_len = -2
                if "重大" in l.group(0):
                    basic_len -= 1
                category = "".join(seg_list[basic_len-1:])
                product_category.append(category)
                flag = True
                break
        if not flag:
            product_name.append("无")
            product_category.append("无")


        payment_re = u"(一次性|分期)[\u4e00-\u9fa5]{2}保险费"
        r = re.compile(payment_re)
        flag = False
        for line in data:
            if "保险费" in line or "退还" not in line or "豁免" not in line or "续" not in line:
                pay = r.search(line)
                if pay == None or ("一次性" not in line and "分期" not in line):
                    continue
                payment = pay.group(0)
                product_payment.append(payment)
                flag = True
                break
        if flag == False:
            product_payment.append("无")

        term_re = u"[\u4e00-\u9fa5]{4}\\[\\d{4}\\][\u4e00-\u9fa5]*\\d{0,4}[\u53f7]"
        r = re.compile(term_re)
        flag = False
        for line in data:
            if r.search(line):
                product_term.append(r.search(line).group(0))
                flag = True
                break
        if flag == False:
            product_term.append("无")


    # print(product_name)
    # print(len(product_name))
    # print(product_category)
    # print(len(product_category))
    # print(product_payment)
    # print(len(product_payment))
    # print(product_term)
    # print(len(product_term))

    product_info = []
    for i in range(0, 1000):
        info = []
        info.append({"file_name": file_name[i]})
        info.append({"product_category": product_category[i]})
        info.append({"product_name": product_name[i]})
        info.append({"product_payment": product_payment[i]})
        info.append({"product_term": product_term[i]})
        product_info.append(info)

    # for info in product_info:
    #     print(info)

    return product_info


def insurance_match(dataset):
    insurance_min_age = []
    insurance_max_age = []
    insurance_period = []
    insurance_payment = []
    file_name = []

    for data in dataset:

        file_name.append(data[-1])

        age_re = u"投保年龄为[\u4e00-\u9fa5]+至[\u4e00-\u9fa5]+。"
        r = re.compile(age_re)
        flag = False
        for line in data:
            age = r.search(line)
            if age:
                age = age.group(0).split("为")[1].split("至")
                min_age = age[0]
                max_age = age[1].strip("。")
                insurance_max_age.append(min_age)
                insurance_min_age.append(max_age)
                flag = True
                break
        if not flag:
            insurance_max_age.append("无")
            insurance_min_age.append("无")


        period_re = u"保险期间[\u4e00-\u9fa5]+为[\u4e00-\u9fa5]+(，?)(。?)"
        r = re.compile(period_re)
        flag = False
        for line in data:
            period = r.search(line)
            if period:
                period = period.group(0).split("为")[1].strip("，").strip("。")
                insurance_period.append(period)
                flag = True
                break
        if not flag:
            insurance_period.append("无")


        payment_re = u"(一次性|分期)[\u4e00-\u9fa5]{2}保险费"
        r = re.compile(payment_re)
        flag = False
        for line in data:
            if "保险费" in line or "退还" not in line or "豁免" not in line or "续" not in line:
                pay = r.search(line)
                if pay == None or ("一次性" not in line and "分期" not in line):
                    continue
                payment = pay.group(0)
                insurance_payment.append(payment)
                flag = True
                break
        if flag == False:
            insurance_payment.append("无")

    insurance_info = []
    for i in range(0, 1000):
        info = []
        info.append({"file_name": file_name[i]})
        info.append({"insurance_min_age": insurance_min_age[i]})
        info.append({"insurance_max_age": insurance_max_age[i]})
        info.append({"insurance_period": insurance_period[i]})
        info.append({"insurance_payment": insurance_payment[i]})
        insurance_info.append(info)

    for info in insurance_info:
        print(info)

    return insurance_info



def write_to_csv(output):
    with open("output.csv", "w", encoding="utf8") as f:
        f.write("保单文件名, 产品类型, 产品名称, 产品交费方式, 产品条款文字编码" + "\n")
        for line in output:
            line = str(line)
            line = line.strip("{").strip("}").strip("[").strip("]").strip("'")
            line = "".join(line)
            # print(line)
            f.write(line + "\n")

# def build_model():
#     model = Sequential()
#     model.add(Dense(units=64, input_dim=13, activation="relu"))
#     model.add(Dense(units=64, input_dim=13))
#     model.add(Dense(1))
#
#     model.compile(loss="mse", optimizer=RMSprop(), metrics=["mae"])
#
#     return model


if __name__ == '__main__':
    dataset = load_data()

    # output = product_match(dataset)
    output = insurance_match(dataset)

    write_to_csv(output)