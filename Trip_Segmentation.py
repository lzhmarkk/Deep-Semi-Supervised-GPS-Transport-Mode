# use pelt, source from https://github.com/ruipgil/changepy
import numpy as np
import time
import pickle
from Instance_Creation import unlabeled_gps_to_trip, compute_trip_motion_features, trip_check_thresholds
from DL_Data_Creation import trip_to_fixed_length

# Change the current working directory to the location of 'Combined Trajectory_Label_Geolife' folder.
# threshold value定义
current = time.perf_counter()
min_threshold = 20
max_threshold = 248
min_distance = 150
min_time = 60
gamma = 1

filename = './Data/Trajectory_Label.pickle'
# labelled结构：[user][point][纬度，经度，时间，交通模式]
# unlabelled结构：[user][point][纬度，经度，时间]
with open(filename, 'rb') as f:
    trajectory_all_user_with_label, trajectory_all_user_wo_label = pickle.load(f)

print(np.array(trajectory_all_user_with_label).shape, np.array(trajectory_all_user_wo_label).shape)

# 调试用
# trajectory_all_user_with_label = trajectory_all_user_with_label[:10]
# trajectory_all_user_wo_label = trajectory_all_user_wo_label[:10]

# Identify the Speed and Acceleration limit
SpeedLimit = {0: 7, 1: 12, 2: 120. / 3.6, 3: 180. / 3.6, 4: 120 / 3.6}
# Online sources for Acc: walk: 1.5 Train 1.15, bus. 1.25 (.2), bike: 2.6, train:1.5
AccLimit = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3}

# trip_all_user_with_label结构：[user][trip][point][纬度，经度，时间，交通模式]
trip_all_user_with_label = [unlabeled_gps_to_trip(trajectory, trip_time=20 * 60) for trajectory in
                            trajectory_all_user_with_label]
print(np.array(trip_all_user_with_label).shape)

# trip_motion_all_user结构：[user][trip][features, mode][value...]
trip_motion_all_user_with_label = [compute_trip_motion_features(user, data_type='custom') for user in
                                   trip_all_user_with_label]
print(np.array(trip_motion_all_user_with_label).shape)

# Apply the threshold values to each GPS segment
# 结构：[user][trip][features, mode][value...]
trip_motion_all_user_with_label = trip_check_thresholds(trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                        min_distance=min_distance, min_time=min_time,
                                                        data_type='labeled')
print(np.array(trip_motion_all_user_with_label).shape)

# 结构：[trip][features, mode][value...]
trip_motion_with_label = [trip for user in trip_motion_all_user_with_label for trip in user]
print(np.array(trip_motion_with_label).shape)

# trip_segments结构：[trip][features, mode][value...]，其中value的长度被fixed为max_length
trip_segments_with_label = trip_to_fixed_length(trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                max_threshold=max_threshold, min_distance=min_distance,
                                                min_time=min_time,
                                                data_type='unlabeled')
print(np.array(trip_segments_with_label).shape)

# use only speed and acceleration feature, according to thesis
# trip_segments.shape = len * 2 * 248
trip_segments = trip_segments_with_label[:, 3:5, :]
# trip_segments_label.shape = len * 248
trip_segments_label = trip_segments_with_label[:, 7]
print(np.array(trip_segments).shape, np.array(trip_segments_label).shape)

# change shape of trip_segments
# trip_segments_new结构：[trip][value...][feature]，其中value的长度被fixed为max_length
trip_segments_new = np.zeros((len(trip_segments), max_threshold, 2))
for i in range(len(trip_segments)):
    trip_segments_new[i, :, 0] = trip_segments[i, 0, :]
    trip_segments_new[i, :, 1] = trip_segments[i, 1, :]
print(np.array(trip_segments_new).shape)

# 找到每个trip的ground truth中的change point
change_point_truth = []
for index, trip in enumerate(trip_segments_label):
    i = 0
    change_point_one_trip = []
    while i < len(trip) - 1:
        mode1 = trip[i]
        mode2 = trip[i + 1]
        if mode1 != mode2:
            change_point_one_trip.append(i)
    change_point_truth.append(change_point_one_trip)
print(np.array(change_point_truth).shape)

# This pickling and unpickling is due to large computation time before this line
filename = './Data/change_points.pickle'
with open(filename, 'wb') as f:
    pickle.dump([trip_segments_new, change_point_truth], f)

with open(filename, 'rb') as f:
    trip_segments_new, change_point_truth = pickle.load(f)


def my_pelt_cost(trip):
    """ Creates a segment cost function for a time series with the following equation
    \\Sigma ||Y_t - Y_bar||^2, Y_bar = mean(Y_t)
    Args:
        trip (:obj:`list` of [value...][feature]): time series data
        trip.shape = 248 * 2
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    if not isinstance(trip, np.ndarray):
        trip = np.array(trip)

    def cost(start, end):
        """ Cost function for normal distribution with variable mean
        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        series = trip[start:end]
        # y_mean.shape = 2
        series_mean = np.mean(series, axis=0)

        sigma = 0
        for i in range(len(series)):
            sigma += np.sum(np.square(series[i] - series_mean))
        return sigma / len(series)

    return cost


def pelt(cost, length, pen=None):
    """ PELT algorithm to compute changepoints in time series

    Ported from:
        https://github.com/STOR-i/Changepoints.jl
        https://github.com/rkillick/changepoint/
    Reference:
        Killick R, Fearnhead P, Eckley IA (2012) Optimal detection
            of changepoints with a linear computational cost, JASA
            107(500), 1590-1598

    Args:
        cost (function): cost function, with the following signature,
            (int, int) -> float
            where the parameters are the start index, and the second
            the last index of the segment to compute the cost.
        length (int): Data size
        pen (float, optional): defaults to log(n)
    Returns:
        (:obj:`list` of int): List with the indexes of changepoints
    """

    def find_min(arr, val=0.0):
        """ Finds the minimum value and index

        Args:
            arr (np.array)
            val (float, optional): value to add
        Returns:
            (float, int): minimum value and index
        """
        return min(arr) + val, np.argmin(arr)

    # this penalty correct?
    if pen is None:
        pen = gamma * length

    F = np.zeros(length + 1)
    R = np.array([0], dtype=np.int)
    candidates = np.zeros(length + 1, dtype=np.int)

    F[0] = -pen

    for tstar in range(2, length + 1):
        cpt_cands = R
        seg_costs = np.zeros(len(cpt_cands))
        for i in range(0, len(cpt_cands)):
            seg_costs[i] = cost(cpt_cands[i], tstar)

        F_cost = F[cpt_cands] + seg_costs
        F[tstar], tau = find_min(F_cost, pen)
        candidates[tstar] = cpt_cands[tau]

        ineq_prune = [val < F[tstar] for val in F_cost]
        R = [cpt_cands[j] for j, val in enumerate(ineq_prune) if val]
        R.append(tstar - 1)
        R = np.array(R, dtype=np.int)

    last = candidates[-1]
    changepoints = [last]
    while last > 0:
        last = candidates[last]
        changepoints.append(last)

    return sorted(changepoints)


# run pelt segmentation
mae_all_trip = []
pr_all_trip = []
for index, trip in enumerate(trip_segments_new):
    change_point_one_trip = pelt(my_pelt_cost(trip=trip), len(trip))
    mae_trip = []
    pr_trip = []
    for pred_cp in change_point_one_trip:
        # find the nearest ground-truth change point of pred_cp in change_point_truth[index]
        def find_nearest(point, array):
            i = 0
            for i in range(len(array)):
                if array[i] == point:
                    return i
                elif array[i] > point:
                    break
            if i >= 1:
                left = abs(point - array[i - 1])
                right = abs(array[i] - point)
                return i - 1 if left < right else i


        mae_trip.append(find_nearest(point=pred_cp, array=change_point_truth[index]))
        pr_trip.append(len(change_point_one_trip) / len(change_point_truth[index]))
    mae_all_trip.append(np.mean(np.array(mae_trip)))
    pr_all_trip.append(np.mean(np.array(pr_trip)))

mae = np.mean(np.array(mae_all_trip))
pr = np.mean(np.array(pr_all_trip))

print("mae %f", mae)
print("pr %f", pr)
