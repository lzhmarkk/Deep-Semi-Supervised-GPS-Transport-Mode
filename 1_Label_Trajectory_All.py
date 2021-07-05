# 准备轨迹数据并将它们与其关联的标签文件进行匹配，以及去除错误和异常值的预处理步骤
import pickle
from datetime import datetime
import os
import time

# py3.8不再支持time.clock()
# start_time = time.clock()
start_time = time.perf_counter()


def days_date(time_str):
    """
    计算time_str相对某个标准时间点过了多少天
    """
    date_format = "%Y/%m/%d %H:%M:%S"
    current = datetime.strptime(time_str, date_format)
    date_format = "%Y/%m/%d"
    bench = datetime.strptime('1899/12/30', date_format)
    no_days = current - bench
    delta_time_days = no_days.days + current.hour / 24.0 + current.minute / (24. * 60.) + current.second / (24. * 3600.)
    return delta_time_days


# Change Mode Name to Mode index
Mode_Index = {"walk": 0, "run": 9, "bike": 1, "bus": 2, "car": 3, "taxi": 3, "subway": 4, "railway": 4,
              "train": 4, "motocycle": 8, "boat": 9, "airplane": 9, "other": 9}

# Ground modes are the modes that we use in this paper.
Ground_Mode = ['walk', 'bike', 'bus', 'car', 'taxi', 'subway', 'railway', 'train']

# 读取Geolife Trajectories 1.3数据
geolife_dir = './Geolife Trajectories 1.3/Data/'
users_folder = os.listdir(geolife_dir)
# unlabelled数据，格式[用户][记录][纬度，经度，时间(以1899/12/30为起始)]
trajectory_all_user_wo_label = []
# labelled数据的轨迹，格式[用户][记录][纬度，经度，时间]
trajectory_all_user_with_label = []
# labelled数据的label，格式[用户][记录][开始时间，结束时间，交通模式]
label_all_user = []

# 读取数据
print("开始读取数据")
for folder in users_folder:
    # 忽略MacOS中的.DS_Store
    if folder == ".DS_Store":
        continue
    # 只有一个子文件，该folder是unlabeled
    if len(os.listdir(geolife_dir + folder)) == 1:
        trajectory_dir = geolife_dir + folder + '/Trajectory/'
        user_trajectories = os.listdir(trajectory_dir)
        trajectory_one_user = []
        # 读取每个Trajectory文件夹内的plt文件
        for plt in user_trajectories:
            with open(trajectory_dir + plt, 'r', newline='', encoding='utf-8') as f:
                # 筛选出有效数据行
                GPS_logs = filter(lambda x: len(x.split(',')) == 7, f)
                # 用map函数，删除末尾换行，并拆分数据行
                GPS_logs_split = map(lambda x: x.rstrip('\r\n').split(','), GPS_logs)
                # 数据行中提取纬度、精度和时间
                for row in GPS_logs_split:
                    trajectory_one_user.append([float(row[0]), float(row[1]), float(row[4])])
        trajectory_all_user_wo_label.append(trajectory_one_user)

    # 有两个子文件，该folder是labeled
    elif len(os.listdir(geolife_dir + folder)) == 2:
        trajectory_dir = geolife_dir + folder + '/Trajectory/'
        user_trajectories = os.listdir(trajectory_dir)
        trajectory_one_user = []
        for plt in user_trajectories:
            with open(trajectory_dir + plt, 'r', newline='', encoding='utf-8') as f:
                GPS_logs = filter(lambda x: len(x.split(',')) == 7, f)
                GPS_logs_split = map(lambda x: x.rstrip('\r\n').split(','), GPS_logs)
                for row in GPS_logs_split:
                    trajectory_one_user.append([float(row[0]), float(row[1]), float(row[4])])
        trajectory_all_user_with_label.append(trajectory_one_user)

        label_dir = geolife_dir + folder + '/labels.txt'
        with open(label_dir, 'r', newline='', encoding='utf-8') as f:
            # 用map函数，删除末尾换行，并拆分数据行
            label = list(map(lambda x: x.rstrip('\r\n').split('\t'), f))
            # 筛选出符合条件的数据行（长度为3，且最后一项为某种交通模式）
            label_filter = list(filter(lambda x: len(x) == 3 and x[2] in Ground_Mode, label))
            label_one_user = []
            # 数据行中提取开始时间、结束时间和交通模式
            for row in label_filter:
                label_one_user.append([days_date(row[0]), days_date(row[1]), Mode_Index[row[2]]])
        label_all_user.append(label_one_user)

# labelled数据，格式[用户][记录][纬度，经度，时间，交通模式]
trajectory_all_user_with_label_Final = []  # Only contain users' trajectories that have labels
for index, user in enumerate(label_all_user):
    # 遍历每个user
    trajectory_user = trajectory_all_user_with_label[index]
    classes = {0: [], 1: [], 2: [], 3: [], 4: []}
    start_index = 0
    for row in user:
        # 遍历每个user的每条label记录
        # 根据其开始时间和结束时间，从trajectory_user从提取出对应时间窗口内的记录
        # 将其index记录在classes中，并且删掉trajectory_user中已经被提取的部分
        if not trajectory_user:
            break

        start = row[0]
        end = row[1]
        mode = row[2]

        if trajectory_user[0][2] > end or trajectory_user[-1][2] < start:
            continue
        for index1, trajectory in enumerate(trajectory_user):
            if start <= trajectory[2] <= end:
                # start_index记录了一个user的trajectory中，第一个能用的记录的index
                start_index += index1
                # 删掉trajectory_user中已经被提取的部分
                trajectory_user = trajectory_user[index1 + 1:]
                break

        if trajectory_user[-1][2] < end:
            # 说明后面所有的trajectory都是这个label的
            end_index = start_index - 1 + len(trajectory_user)
            # 记录trajectory记录的index
            classes[mode].extend(list(range(start_index, end_index)))
            break
        else:
            # 此处的trajectory_user已经被上面裁剪过
            for index2, trajectory in enumerate(trajectory_user):
                # 找到对应时间窗口内的记录
                if trajectory[2] > end:
                    end_index = start_index + 1 + index2
                    # 删掉trajectory_user中已经被提取的部分
                    trajectory_user = trajectory_user[index2:]
                    # 记录trajectory记录的index
                    classes[mode].extend(list(range(start_index, end_index)))
                    start_index = end_index
                    break

    # k代表交通模式的index，v代表对应的trajectory的index的集合
    for k, v in classes.items():
        for value in v:
            # labelled数据的轨迹，格式更新为[用户][记录][纬度，经度，时间，交通模式]
            trajectory_all_user_with_label[index][value].append(k)

    # 去掉缺失的数据
    labeled_trajectory = list(filter(lambda x: len(x) == 4, trajectory_all_user_with_label[index]))
    trajectory_all_user_with_label_Final.append(labeled_trajectory)
    unlabeled_trajectory = list(filter(lambda x: len(x) == 3, trajectory_all_user_with_label[index]))
    trajectory_all_user_wo_label.append(unlabeled_trajectory)
    print("labelled user %d done" % index)

# Save Trajectory_Array and Label_Array for all users
try:
    os.makedirs("./Data/")
except:
    pass
with open("./Data/Trajectory_Label.pickle", 'wb') as f:
    pickle.dump([trajectory_all_user_with_label_Final, trajectory_all_user_wo_label], f)

print("预处理用时", time.perf_counter() - start_time, 'Seconds')
