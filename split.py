import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
from sklearn.preprocessing import RobustScaler

def _gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-u ** 2 / 2)

def _kde(data, grid_points=500):
    data = np.asarray(data).flatten()
    n = len(data)
    EPS=1e-10
    if n == 0:
        return np.array([]), np.array([])
    x = np.linspace(np.min(data), np.max(data), grid_points)
    sigma = np.std(data)
    h = ((4 / (3 * n)) ** (1 / 5)) * (sigma if sigma > EPS else EPS)
    f_x_all = []
    for xi in x:
        f_x = (1 / (n * h)) * np.sum(_gaussian_kernel((xi - data) / h))
        f_x_all.append(f_x)
    return x, np.array(f_x_all)

def AD_test_judge(x_p):
    x_p = np.asarray(x_p).flatten()
    EPS = 1e-10
    size_ = len(x_p)
    if size_ < 3:
        return 1

    std_ =  np.std(x_p)
    if std_ < EPS:
        std_ = EPS
    x_p = (x_p - np.mean(x_p)) / std_
    z_p = np.sort(x_p)
    z_i = norm.cdf(z_p)

    AD = 0
    for i in range(size_):
        AD_1 = 2 * (i + 1) - 1
        AD_2 = np.log(z_i[i]+EPS)
        AD_3 = np.log(1 - z_i[size_ - 1 - i] + EPS)
        AD += AD_1 * (AD_2 + AD_3)
    ADZ = -AD / size_ - size_
    ADZ_m = ADZ * (1 + 4 / size_ - 25 / (size_) ** 2)
    if ADZ_m > 1.8692:
        return 0
    else:
        return 1
def adjacency_similarity(X_1, X_2):
    X_m = np.concatenate([X_1, X_2])
    EPS = 1e-10
    c1 = np.mean(X_1, axis=0)
    c2 = np.mean(X_2, axis=0)
    v_i = c1 - c2
    v_i = v_i.reshape(1,-1)
    denom =  (v_i @ v_i.T)

    if denom < EPS:
        return 1.0
    x_p_i = (((X_m - c2) @ v_i.T) /denom)
    smooth_p = _kde(x_p_i.flatten())
    x_grid = smooth_p[0]
    density = smooth_p[1]

    mask = (x_grid >= 0) & (x_grid <= 1)
    density_region = density[mask]

    denom_conn =  density_region[0] + density_region[-1]
    if denom_conn < EPS:
        return 0.0
    connection = (np.min(density_region) * 2) /denom_conn

    return connection

def linearity_similarity(X_1, X_2):
    if X_1.shape[0] < 3:
        return 0

    elif X_2.shape[0] < 3:
        return 0

    else:
        convex_hull_C1 = ConvexHull(X_1)
        convex_hull_C2 = ConvexHull(X_2)

        C1_area = convex_hull_C1.volume
        C1_vertice = convex_hull_C1.vertices
        C2_area = convex_hull_C2.volume
        C2_vertice = convex_hull_C2.vertices

        vertices_all = np.concatenate([X_1[C1_vertice], X_2[C2_vertice]])
        convex_hull_all = ConvexHull(vertices_all)
        Call_area = convex_hull_all.volume
        convex_connection = ((np.min([C1_area, C2_area])) / (
                Call_area - np.max([C1_area, C2_area]) + 1e-10))
        return convex_connection
def split_judge(X_1, X_2, p_thr=0.5, c_thr=0.5):
    p_judge = adjacency_similarity(X_1, X_2)
    if p_judge < p_thr:
        return 0
    else:
        c_judge = linearity_similarity(X_1, X_2)
        if c_judge < c_thr:
            return 0
        else:
            return 1
def split(X, y, p_thr=0.5, c_thr=0.5, max_iters=20, k=2, seed = None):
    beta_hat = np.linalg.inv(X.T@X)@X.T@y    ##OLS estimation
    y_hat = X@beta_hat
    r = y-y_hat
    X_2d = np.concatenate([r, y_hat], axis=1) ##2-D data construction
    scaler = RobustScaler()
    X_2d = scaler.fit_transform(X_2d)

    size_ = X_2d.shape[0]

    iter_continue = True
    iter_count = 0
    labels = np.zeros(size_, dtype=int)
    C = np.zeros((k, 2))
    while iter_continue:
        iter_count += 1
        if iter_count > max_iters:
            break

        gmm = GaussianMixture(n_components=k,random_state=seed)
        gmm.fit(X_2d)
        labels = gmm.predict(X_2d)
        C = gmm.means_
        new_add_center = np.ones((1, X_2d.shape[1]))
        pre_delete_location = []

        for i in list(set(labels)):
            X_i = X_2d[labels == i]
            C_i = np.empty((0, 2))

            if X_i.shape[0] < 2:
                is_same = 1
            else:
                gmm_i = GaussianMixture(n_components=2,random_state=seed)
                gmm_i.fit(X_i)
                labels_i = gmm_i.predict(X_i)
                C_i = gmm_i.means_
                v_i = C_i[1] - C_i[0]
                v_i = v_i.reshape(1, 2)
                x_p_i = (((X_i - C_i[0]) @ v_i.T) / (v_i @ v_i.T)) ##projection
                is_normal_i = AD_test_judge(x_p_i.flatten()) ##namality criterion

                if is_normal_i == 1:
                    is_same = 1
                else:
                    unique_labels = list(set(labels_i))
                    if len(unique_labels) == 1:
                        is_same = 1
                    else:
                        X_1 = X_i[labels_i == unique_labels[0]]
                        X_2 = X_i[labels_i == unique_labels[1]]
                        is_same = split_judge(X_1, X_2, p_thr, c_thr) ##adjacency and linearity criteria

            if is_same == 0 and C_i.size > 0:
                new_add_center = np.concatenate([new_add_center, C_i])
                pre_delete_location.append(i)

        if new_add_center.shape[0] == 1:
            iter_continue = False
        else:
            new_add_center = new_add_center[1:]
            if pre_delete_location:
                C = np.delete(C, pre_delete_location, axis=0)
            C = np.concatenate([C, new_add_center])
        k = C.shape[0]

    labels_num = 0
    C_new = np.zeros((len(set(labels)), 2))
    labels_new = np.zeros((size_,)) - 1
    for i in set(labels):
        labels_new[labels == i] = int(labels_num)
        C_new[labels_num] = C[i]
        labels_num += 1
    labels = labels_new.astype(int)

    return labels
