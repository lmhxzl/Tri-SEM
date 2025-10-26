import numpy as np
from scipy.stats import t

def _find_median_position(data):
    data = np.array(data)
    median_value = np.median(data)
    median_indices = np.where(data == median_value)[0].tolist()
    if len(median_indices) == 0:
        sorted_data = np.sort(data)
        n = len(sorted_data)
        if n % 2 == 0:
            lower_median = sorted_data[n // 2 - 1]
            upper_median = sorted_data[n // 2]
            median_indices = np.where((data == lower_median) | (data == upper_median))[0].tolist()
    return median_value, median_indices
def _htd(data):
    data = np.asarray(data).flatten()
    if data.size < 3:
        return np.array([np.nan] * data.size)
    curvature = []
    for i in range(1, len(data) - 1):
        left = np.mean(data[:i])
        right = np.mean(data[(i + 1):])
        data_i = np.mean(data[(i - 1):(i + 2)])
        curvature.append(((left + right - 2 * data_i)))
    curvature = np.array(curvature)
    return np.concatenate(([np.nan], curvature, [np.nan]))

def merge(X, y, labels,partial_clean_label, beta_robust, h_thr =0.5):
    size_, p = X.shape
    r_f = y - X @ beta_robust
    hist, bins = np.histogram(np.abs(r_f), bins=int(2 * np.sqrt(size_)))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    curvature = _htd(hist)
    max_curv_idx = np.nanargmax(curvature)
    max_curv = np.nanmax(curvature)
    cur = curvature[max_curv_idx:][:-1]
    vp = np.median(cur[-int(np.ceil(len(cur) * 0.5)):])
    judge_upper = ((max_curv - vp) / max_curv) >= h_thr ##Prominence criterion
    if judge_upper:
        upper_bound = bin_centers[np.nanargmax(curvature)]
    else:##Hypothesis test criterion
        clean_choose = [i for i in range(size_) if np.abs(r_f[i]) < bin_centers[np.nanargmax(curvature)]]
        xm = X[clean_choose]
        ym = y[clean_choose]
        beta_h = np.linalg.inv(xm.T @ xm) @ xm.T @ ym
        r_h = ym - xm @ beta_h
        sigma_judge = (r_h.T @ r_h) / (len(clean_choose) - p)
        sigma_judge = np.sqrt(sigma_judge[0][0])
        outlier_choose = [i for i in range(size_) if np.abs(r_f[i]) >= bin_centers[max_curv_idx]]
        judge_location = outlier_choose[_find_median_position(np.abs(r_f[outlier_choose]))[1][0]]
        xi = X[judge_location]
        yi = y[judge_location]
        hi = xi.T @ np.linalg.inv(xm.T @ xm) @ xi
        ri = yi - xi.T @ beta_h
        di = ri / (sigma_judge * np.sqrt(1 + hi))
        di = di[0]
        alpha = 0.05
        t_critical_two_sided = t.ppf(1 - (alpha / (2 * (len(clean_choose) + 1))), len(clean_choose) - p)
        if np.abs(di) > t_critical_two_sided:
            upper_bound = bin_centers[np.nanargmax(curvature)]
        else:
            upper_bound = np.inf

    choosen_location = []

    for i in range(size_):
        r_fi = np.abs(r_f[i])
        if r_fi < upper_bound:
            choosen_location.append(i)

    choosen_location = np.array(choosen_location)

    # refinement
    unique_labels = np.unique(labels[choosen_location])
    new_clean_label = np.array([item for item in unique_labels if item not in partial_clean_label])

    new_clean_label_final = []
    for item in new_clean_label:
        choosen_sum = np.sum(labels[choosen_location] == item)
        all_sum = np.sum(labels == item)
        if choosen_sum >= 0.8 * all_sum:
            new_clean_label_final.append(item)

    new_clean_label_final = np.array(new_clean_label_final)

    label_choosen_final = np.concatenate([partial_clean_label, new_clean_label_final])
    label_choosen_final = label_choosen_final.astype(int)

    all_clean_location = np.isin(labels, label_choosen_final)

    X_clean = X[all_clean_location]
    y_clean = y[all_clean_location]
    beta_robust = np.linalg.inv(X_clean.T@X_clean)@X_clean.T@y_clean
    return beta_robust
