import pickle

# 打开pickle文件并加载数据
with open('bbox_list.pickle', 'rb') as file:
    data = pickle.load(file)

# 查看数据
print(data)