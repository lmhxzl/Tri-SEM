import numpy as np
from itertools import combinations

def extraction(X, y, labels):
    size_= X.shape[0]
    unique_labels, num_cluster = np.unique(labels, return_counts=True)
    max_size = np.max(num_cluster)

    if max_size / size_ >= 0.5: ##size_based selction
        max_label = unique_labels[np.argmax(num_cluster)]
        choose_location = labels == max_label
        X_f = X[choose_location]
        y_f = y[choose_location]
        beta_robust = np.linalg.inv(X_f.T @ X_f) @ X_f.T @ y_f
        partial_clean_label= [max_label]

    else:##LMS_based selction
        two_combinations = list(combinations(range(len(unique_labels)), 2))
        combinations_list = np.array(two_combinations)

        theta = []
        median_r2 = []
        for label in combinations_list:
            location_now = np.array([labels[i] in label for i in range(size_)])
            X_f = X[location_now]
            y_f = y[location_now]
            rank_matrix = X_f.T @ X_f
            if np.linalg.matrix_rank(rank_matrix) < np.min([rank_matrix.shape[0], rank_matrix.shape[1]]):
                theta.append(np.nan)
                median_r2.append(np.inf)
            else:
                beta_now = np.linalg.inv(X_f.T @ X_f) @ \
                           X_f.T @ y_f
                theta.append(beta_now)
                r_now = y - X @ beta_now
                m = np.median(r_now.flatten() ** 2)
                median_r2.append(m)

        location_robust = np.argmin(median_r2)
        beta_robust = theta[location_robust]
        partial_clean_label = combinations_list[location_robust]

    return partial_clean_label, beta_robust ##Clean data labels and the OLS estimation on clean data
