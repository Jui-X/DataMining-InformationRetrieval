import json


def load_data():
    user_seq = json.load(open('data/user_seq.json', 'r'))
    return user_seq


if __name__ == '__main__':
    user_seq = load_data()
    for seq in user_seq:
        print(seq)