# use pelt to segment trips
import numpy as np
import time
import pickle
from Instance_Creation import unlabeled_gps_to_trip, compute_trip_motion_features
from DL_Data_Creation import trip_to_fixed_length
import ruptures
from ruptures.costs import costl2, NotEnoughPoints
import math

# threshold value定义
min_threshold = 20
max_threshold = 248
min_distance = 150
min_time = 60
gamma = 650  # [0,4,8,14,16,20]


class my_l2_cost(costl2.CostL2):
    def __init__(self):
        super(my_l2_cost, self).__init__()

    def error(self, start, end) -> float:
        if end - start < self.min_size:
            raise NotEnoughPoints
        return self.signal[start:end].var(axis=0).sum() * (end - start) + gamma * (end - start)


def trip_check_thresholds_label(trip_motion_all_user, min_threshold, min_distance, min_time):
    # Remove trip with less than a min GPS point, less than a min-distance, less than a min trip time.
    all_user = []
    for user in trip_motion_all_user:
        all_user.append(
            list(filter(lambda trip: len(trip[0]) >= min_threshold and np.sum(trip[0, :]) >= min_distance
                                     and np.sum(trip[1, :]) >= min_time, user)))
    return all_user


def load(step):
    with open('./Data/temp' + str(step) + '.pickle', 'rb') as f:
        data = pickle.load(f)
    print("data loaded")
    return data


if __name__ == '__main__':
    print("running Trip_Segmentation")
    current = time.perf_counter()
    # 如果change points还没被计算出来，就要根据Trajectory_Label.pickle的轨迹信息算一遍

    print('change point data not exists, generating')
    """##########   step1   ##########"""
    trajectory_file = './Data/Trajectory_Label.pickle'
    # labelled结构：[user][point][纬度，经度，时间，交通模式]
    # unlabelled结构：[user][point][纬度，经度，时间]
    with open(trajectory_file, 'rb') as f:
        trajectory_all_user_with_label, _ = pickle.load(f)
    print("step1", time.perf_counter() - current)

    """##########   step2   ##########"""
    # Identify the Speed and Acceleration limit
    SpeedLimit = {0: 7, 1: 12, 2: 120. / 3.6, 3: 180. / 3.6, 4: 120 / 3.6}
    # Online sources for Acc: walk: 1.5 Train 1.15, bus. 1.25 (.2), bike: 2.6, train:1.5
    AccLimit = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3}
    # trip_all_user_with_label结构：[user][trip][point][纬度，经度，时间，交通模式]
    trip_all_user_with_label = [unlabeled_gps_to_trip(trajectory, trip_time=20 * 60) for trajectory in
                                trajectory_all_user_with_label]
    print("step2", time.perf_counter() - current)

    """##########   step3   ##########"""
    # trip_motion_all_user结构：[user][trip][features, mode][value...]
    trip_motion_all_user_with_label = [compute_trip_motion_features(user, data_type='custom') for user in
                                       trip_all_user_with_label]
    print("step3", time.perf_counter() - current)

    """##########   step4   ##########"""
    # Apply the threshold values to each GPS segment
    # 结构：[user][trip][features, mode][value...]
    trip_motion_all_user_with_label = trip_check_thresholds_label(trip_motion_all_user_with_label,
                                                                  min_threshold=min_threshold,
                                                                  min_distance=min_distance, min_time=min_time)
    print("step4", time.perf_counter() - current)

    """##########   step5   ##########"""
    # 结构：[trip][features, mode][value...]，其中value的长度不一
    trip_motion_with_label = [trip for user in trip_motion_all_user_with_label for trip in user]
    print("step5", time.perf_counter() - current)

    """##########   step6   ##########"""
    # trip_segments结构：[trip][features, mode][value...]，其中value的长度被fixed为max_length
    trip_segments_with_label = trip_to_fixed_length(trip_motion_with_label, min_threshold=min_threshold,
                                                    max_threshold=max_threshold, min_distance=min_distance,
                                                    min_time=min_time,
                                                    data_type='unlabeled')
    print("step6", time.perf_counter() - current)

    """##########   step7   ##########"""
    # use only speed and acceleration feature, according to thesis
    # trip_segments.shape = len * 2 * 248
    trip_segments = trip_segments_with_label[:, 3:5, :]
    # trip_segments_label.shape = len * 248
    trip_segments_label = trip_segments_with_label[:, 7]
    print("step7", time.perf_counter() - current)

    """##########   step8   ##########"""
    # change shape of trip_segments
    # trip_segments_new结构：[trip][value...][feature]，其中value的长度被fixed为max_length
    trip_segments_new = np.zeros((len(trip_segments), max_threshold, 2))
    for i in range(len(trip_segments)):
        trip_segments_new[i, :, 0] = trip_segments[i, 0, :]
        trip_segments_new[i, :, 1] = trip_segments[i, 1, :]
    print("step8", time.perf_counter() - current)

    """##########   step9   ##########"""
    # 找到每个trip的ground truth中的change point
    change_point_truth = []
    for index, trip in enumerate(trip_segments_label):
        i = 0
        change_point_one_trip = []
        for i in range(len(trip) - 1):
            mode1 = trip[i]
            mode2 = trip[i + 1]
            if mode1 != mode2:
                change_point_one_trip.append(i)
        change_point_truth.append(change_point_one_trip)
    print("step9", time.perf_counter() - current)

    with open('./Data/change_points.pickle', 'wb') as f:
        pickle.dump([trip_segments, change_point_truth], f)
        print("计算change points完成")

    """##########   step10   ##########"""
    # 运行PELT算法，预测change points
    mae_all_trip = []
    change_point_pred = []
    for index, trip in enumerate(trip_segments_new):
        # trip.shape = 248 * 2(features)
        algo = ruptures.Pelt(model='l2', jump=1, min_size=min_threshold).fit(trip)
        # algo = ruptures.Pelt(custom_cost=my_l2_cost, jump=1, min_size=min_threshold).fit(trip)
        change_point_one_trip = algo.predict(pen=gamma)
        change_point_one_trip.pop()
        change_point_pred.append(change_point_one_trip)

        mae_trip = []
        for pred_cp in change_point_one_trip:
            def find_nearest(point, arr_list, i):
                min_dist = math.inf
                for p in arr_list[i]:
                    min_dist = min(abs(point - p), min_dist)
                return min(min_dist, abs(point - 0), abs(point - 247))


            mae_trip.append(find_nearest(pred_cp, change_point_truth, index))

        mae_all_trip.append(np.mean(np.array(mae_trip)) if mae_trip else 0)

    """##########   step11   ##########"""
    cp_all_trip_truth = 0
    cp_all_trip_pred = 0
    for change_point_one_trip in change_point_truth:
        cp_all_trip_truth += len(change_point_one_trip)
    for change_point_one_trip in change_point_pred:
        cp_all_trip_pred += len(change_point_one_trip)

    # debug
    for i in range(len(change_point_truth)):
        print(change_point_truth[i], change_point_pred[i])
    print("\ncp_all_trip_pred", cp_all_trip_pred)
    print("cp_all_trip_truth", cp_all_trip_truth)
    print("mse_all_trip", np.mean(mae_all_trip))
    # PR = 预测的cp个数 / 真实的cp个数
    pr = cp_all_trip_pred / cp_all_trip_truth
    # MAE = mean|预测的cp位置 - 最近的真实cp位置| = mean|预测的cp位置 - 最近真实的cp位置或者trip边界| + 两个真实CP间平均距离
    mae = np.mean(mae_all_trip) + max_threshold / (cp_all_trip_pred / len(trip_segments_new))

    print("time", time.perf_counter() - current)
    print("mae", mae)
    print("pr", pr)
