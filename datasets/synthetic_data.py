import numpy as np
import random

def generate_scenario1(n, p, contam_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)

    beta = np.zeros((p + 1, 1)) + 5
    contam_num = int(n * contam_rate)
    clean_num = n - contam_num
    half_clean = int(clean_num / 2)

    # Clean data
    x_clean = np.ones((clean_num, p + 1))
    x_clean[:half_clean, 1:] = 5 + 2 * np.random.randn(half_clean, p)
    x_clean[half_clean:, 1:] = np.random.uniform(low=-10, high=-9, size=(clean_num - half_clean, p))
    e_clean = np.random.standard_t(3, size=(clean_num, 1))
    y_clean = x_clean @ beta + e_clean

    # Contaminated data
    x_contam = np.ones((contam_num, p + 1))
    x_contam[:, 1:] = np.random.uniform(20, 22, size=(contam_num, p))
    x_pretend = np.ones((contam_num, p + 1))
    x_pretend[:, 1:] = np.random.uniform(-22, -20, size=(contam_num, p))
    e_contam = np.random.standard_t(3, size=(contam_num, 1))
    y_contam = x_pretend @ beta + e_contam

    # Combine
    X = np.concatenate([x_clean, x_contam])
    y = np.concatenate([y_clean, y_contam])
    clean_location = np.array(range(clean_num))


    return X, y,clean_location


def generate_scenario2(n, p, contam_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)

    beta = np.zeros((p + 1, 1)) + 5

    contam_num = int(n * contam_rate)
    clean_num = n-contam_num

    three_clean = int(clean_num/3)


    # Clean data
    x_clean = np.ones((clean_num, p+1))
    x_clean[:three_clean,1:] = np.random.uniform(low=0, high=3, size=(three_clean, p))
    x_clean[three_clean:(2*three_clean),1:] = np.random.uniform(low=6, high=9, size=(three_clean, p))
    x_clean[(2*three_clean):,1:] = np.random.uniform(low=12, high=15, size=(clean_num-2*three_clean, p))

    e_clean = np.random.randn(clean_num,1)
    y_clean = x_clean @ beta + e_clean

    # Contaminated data
    x_contam = np.ones((contam_num, p+1))
    x_contam[:,1:] =  np.random.uniform(14, 15, size=(contam_num, p))
    x_pretend = np.ones((contam_num, p+1))
    x_pretend[:, 1:] = np.random.uniform(15, 16, size=(contam_num, p))

    e_contam = np.random.randn(contam_num,1)
    y_contam = x_pretend @ beta + e_contam

    # Combine
    X = np.concatenate([x_clean, x_contam])
    y = np.concatenate([y_clean, y_contam])
    clean_location = np.array(range(clean_num))


    return X, y,clean_location


def generate_scenario3(n, p, contam_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)

    beta = np.zeros((p + 1, 1)) + 5

    contam_num = int(n * contam_rate)
    clean_num = n - contam_num

    # Clean data
    x_clean = np.ones((clean_num, p+1))
    x_clean[:, 1:] = np.random.uniform(low=0, high=1, size=(clean_num, p))
    e_clean = np.random.rand(clean_num, 1)
    y_clean = x_clean @ beta + e_clean

    # Contaminated data
    x_contam = np.ones((contam_num, p+1))
    x_contam[:, 1:] = np.random.uniform(-4, -2, size=(contam_num, p))
    e_contam = np.random.rand(contam_num, 1)
    beta_contam = np.ones((p+1, 1)) - 6
    beta_contam[0] = 5
    y_contam = x_contam @ beta_contam + e_contam

    # Combine
    X = np.concatenate([x_clean, x_contam])
    y = np.concatenate([y_clean, y_contam])
    clean_location = np.array(range(clean_num))

    return X, y,clean_location

def generate_scenario4(n, p, contam_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    beta = np.zeros((p + 1, 1)) + 5

    contam_num = int(n * contam_rate)
    clean_num = n-contam_num

    # Clean data
    x_clean = np.ones((clean_num, p+1))
    x_clean[:,1:] = 5*np.random.randn(clean_num, p)

    e_clean = 5*np.random.randn(clean_num,1)
    y_clean = x_clean @ beta + e_clean

    # Contaminated data
    noise_random = random.sample(range(clean_num),contam_num)
    x_contam = x_clean[noise_random]
    y_contam = y_clean[noise_random] + (2*np.std(y_clean))*np.random.randn(len(noise_random),1)

    # Combine
    X = np.concatenate([x_clean, x_contam])
    y = np.concatenate([y_clean, y_contam])
    clean_location = np.array(range(clean_num))


    return X, y,clean_location


def generate_scenario5(n, p, seed=None):
    if seed is not None:
        np.random.seed(seed)

    beta = np.zeros((p + 1, 1)) + 5

    # Clean data only
    X = np.ones((n, p+1))
    X[:,1:] = 5+1*np.random.randn(n, p)
    e_clean = np.random.randn(n,1)
    y = X @ beta + e_clean
    clean_location = np.array(range(n))

    return X, y,clean_location
