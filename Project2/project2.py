import xlsxwriter, xlrd
import numpy as np
import math


def data_pre_process():
    raw_data = xlrd.open_workbook("output/output.xlsx")
    raw_data.sheet_names()
    table = raw_data.sheet_by_index(0)
    nrows = table.nrows
    ncols = table.ncols
    data = []
    for i in range(1, nrows):
        row_data = []
        for j in range(1, ncols):
            if table.cell_value(i, j) == "":
                row_data.append(0)
            else:
                row_data.append(int(table.cell_value(i, j)))
        data.append(row_data)

    return data


def write_excel():
    user_id = set()
    product_id = set()
    rating = []
    with open("data/training_set.ss", "r", encoding="utf8") as f:
        training_data = f.readlines()
        for lines in training_data:
            line = lines.strip("\n")
            data = line.split("\t")
            user_id.add(data[0].strip("/"))
            product_id.add(data[1].strip("\\"))
            rating.append({"user": data[0].strip("/"), "product": data[1].strip("\\"), "rating": data[2]})

    workbook = xlsxwriter.Workbook("output/output.xlsx")
    worksheet = workbook.add_worksheet()

    worksheet.set_column(1, len(user_id))

    row = 0
    col = 1
    for user in user_id:
        worksheet.write_string(row, col, user)
        col += 1

    row = 1
    col = 0
    for product in product_id:
        worksheet.write_string(row, col, product)
        row += 1

    workbook.close()

    data = xlrd.open_workbook("output/output.xlsx")
    data.sheet_names()
    table = data.sheet_by_index(0)
    nrows = table.nrows
    ncols = table.ncols

    workbook = xlsxwriter.Workbook("output/output.xlsx")
    worksheet = workbook.add_worksheet()

    row = 0
    col = 1
    for user in user_id:
        worksheet.write_string(row, col, user)
        col += 1

    row = 1
    col = 0
    for product in product_id:
        worksheet.write_string(row, col, product)
        row += 1

    for rate in rating:
        for j in range(ncols):
            user_value = table.cell_value(0, j)
            if rate["user"] == user_value:
                for i in range(nrows):
                    product_value = table.cell_value(i, 0)
                    if rate["product"] == product_value:
                        worksheet.write_string(i, j, rate["rating"])

    workbook.close()


def lfm(data, k):
    data = np.array(data)
    m, n = data.shape

    alpha = 0.01
    lambda_ = 0.01
    u = np.random.rand(m, k)
    v = np.random.randn(k, n)
    for t in range(1000):
        for i in range(m):
            for j in range(n):
                if math.fabs(data[i][j]) > 1e-4:
                    err = data[i][j] - np.dot(u[i], v[:, j])
                    for r in range(k):
                        gu = err * v[r][j] - lambda_ * u[i][r]
                        gv = err * u[i][r] - lambda_ * v[r][j]
                        u[i][r] += alpha * gu
                        v[r][j] += alpha * gv
    return np.dot(u, v)


def validate(predict):
    user_id = []
    product_id = []
    rating = []
    with open("data/validation_set.ss", "r", encoding="utf8") as f:
        training_data = f.readlines()
        for lines in training_data:
            line = lines.strip("\n")
            data = line.split("\t")
            user_id.append(data[0])
            product_id.append(data[1])
            rating.append(data[2])

    del rating[0]

    data = xlrd.open_workbook("output/output.xlsx")
    data.sheet_names()
    table = data.sheet_by_index(0)
    nrows = table.nrows
    ncols = table.ncols
    row = []
    col = []

    for index in range(len(user_id)):
        for j in range(ncols):
            user_value = table.cell_value(0, j)
            if user_id[index] == user_value:
                col.append(index)
    for index in range(len(product_id)):
        for i in range(nrows):
            product_value = table.cell_value(i, 0)
            if product_id[index] == product_value:
                row.append(index)

    # 计算预测矩阵与验证集中对应真实值的均方误差根RSME
    res = 0
    for i in range(len(row)):
        diff = predict[row[i]][col[i]] - int(rating[i])
        res += math.pow(diff, 2)
    res /= len(row)
    res = math.sqrt(res)

    return res


def test(predict):
    user_id = []
    product_id = []
    rating = []
    with open("data/test_set.ss", "r", encoding="utf8") as f:
        training_data = f.readlines()
        for lines in training_data:
            line = lines.strip("\n")
            data = line.split("\t")
            user_id.append(data[0].strip("/"))
            product_id.append(data[1].strip("\\"))

    data = xlrd.open_workbook("output/output.xlsx")
    data.sheet_names()
    table = data.sheet_by_index(0)
    nrows = table.nrows
    ncols = table.ncols
    row = []
    col = []

    for index in range(len(user_id)):
        for j in range(1, ncols):
            user_value = table.cell_value(0, j)
            if user_id[index] == user_value:
                col.append(j-1)
    for index in range(len(product_id)):
        for i in range(1, nrows):
            product_value = table.cell_value(i, 0)
            if product_id[index] == product_value:
                row.append(i-1)

    rating = []
    for i in range(len(row)):
        rate = []
        rate.append(row[i])
        rate.append(col[i])
        rate.append(predict[row[i]][col[i]])
        rating.append(rate)

    with open("output/test_prediction.dat", "w", encoding="utf8") as f:
        for rate in rating:
            f.write('%s\t%s\t%s\n' % (table.cell_value(0, rate[1]+1), table.cell_value(rate[0]+1, 0), rate[2]))

    return rating


if __name__ == '__main__':
    data = np.array((data_pre_process()))
    # cf()
    predict = lfm(data, 1)
    # print(predict)

    res = validate(predict)
    print("RMSE:" + str(res))

    test_res = test(predict)
    # print(test_res)