# from: http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf
def quantile(vals, alpha, epsilon, lower, upper):
    k = len(vals)
    vals = [lower if v < lower else upper if v > upper else v for v in vals]
    vals = sorted(vals)
    Z = [lower] + vals + [upper]
    Z = [-lower + v for v in Z]  # shift right to be 0 bounded
    y = [(Z[i + 1] - Z[i]) * np.exp(-epsilon * np.abs(i - alpha * k)) for i in range(len(Z) - 1)]
    y_sum = sum(y)
    p = [v/y_sum for v in y]
    idx = np.random.choice(range(k+1), 1, False, p)[0]
    v = np.random.uniform(Z[idx], Z[idx+1])
    return v + lower # shift right
