import json


def load_data():
    user_seq = json.load(open('data/user_seq.json', 'r'))
    return user_seq


if __name__ == '__main__':
    user_seq = load_data()
    last_user = ""
    data = {}
    with open("data/train.csv", "r") as f:
        train_data = f.readlines()
        scores = []
        for raw_data in train_data:
            raw = raw_data.split(',')
            scores.append({raw[0]: raw[1].strip("\n")})

    for seq in user_seq:
        print(seq)
    #     if not last_user or last_user != seq["user"]:
    #         last_user = seq["user"]
    #     else:
    #         post = []
    #         post.append({"post_content": seq["text"]})
    #         post.append({"post_time": seq["time"]})
    #         for score in scores:
    #             if seq["id_str"] in score:
    #                 post.append({"score": score[seq["id_str"]]})
    #         data[last_user] = post
    # print(data)