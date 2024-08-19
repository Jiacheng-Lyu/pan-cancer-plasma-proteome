import numpy as np
import scipy.stats
from scipy.stats._stats_py import _chk_asarray


def fdr(pvals, method='indep'):
    '''
    Calculate FDR by statsmodels.stats.multitest.fdrcorrection
    :param pvals: array-like 1d
    :param method:
        'i', 'indep', 'p', 'poscorr': Benjamini/Hochberg for independent or positively correlated tests
        'n', 'negcorr': fdr_by (Benjamini/Yekutieli for general or negatively correlated tests

    return:
     FDR: array-like 1d
    '''

    pvalues = pvals.copy()
    from statsmodels.stats import multitest
    indices = False
    if np.isnan(pvalues).any():
        indices = True
        pvals_indices = np.vstack((range(len(pvalues)), pvalues))
        pvals_indices = pvals_indices[:, ~(
            np.isnan(pvals_indices).any(axis=0))]
        ps = pvals_indices[1]
    else:
        ps = pvalues

    fdr_out = multitest.fdrcorrection(ps, method=method)
    if indices:
        rejected = np.repeat([np.nan], len(pvalues))
        fdr_ = np.repeat([np.nan], len(pvalues))
        np.put(rejected, pvals_indices[0].astype(int), fdr_out[0])
        np.put(fdr_, pvals_indices[0].astype(int), fdr_out[1])
        return (rejected, fdr_)
    else:
        return fdr_out


def obtain_ranked(a):
    if np.all(np.isnan(a)):
        ranked = scipy.stats.rankdata(a, axis=1, nan_policy='omit')
    else:
        # masked_n1 = np.count_nonzero(~np.isnan(a), axis=1)
        mask = a.copy()
        mask[mask == mask] = 1
        mask[mask != mask] = np.nan
        ranked = scipy.stats.rankdata(a, axis=1, nan_policy='omit') * mask
        # ranked[ranked==0] = np.nan
    return ranked


def tiecorrect_vectorized(rankvals):
    global idx, tmp, cnt, size
    arr = np.sort(rankvals, axis=1)
    add_list = np.ones((rankvals.shape[0], 1), dtype=bool)
    tmp = np.nonzero(
        np.concatenate((add_list, arr[:, 1:] != arr[:, :-1]), axis=1))
    idx = np.split(tmp[1], np.cumsum(np.unique(tmp[0],
                                               return_counts=True)[1]))[:-1]

    cnt = np.asarray(
        list(map(lambda x: (x[1:] - x[:-1]).astype(np.float64), idx)), dtype='object')
    size = np.float64(arr[0].size)

    return np.ones(
        (1, rankvals.shape[0])
    ) if size < 2 else 1.0 - np.asarray(list(map(lambda x: np.sum(x), cnt**3 - cnt))) / (size**3 - size)


def chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis


def square_of_sums(a, axis=0):
    a, axis = _chk_asarray(a, axis)
    s = np.nansum(a, axis)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s


def sum_of_squares(a, axis=0):
    a, axis = _chk_asarray(a, axis)
    return np.nansum(a*a, axis)


def ranksums_vectorized(x, y):
    n1 = x.shape[1]
    n2 = y.shape[1]
    alldata = np.hstack((x, y))

    if np.all(np.isnan(y)) & np.all(np.isnan(x)):
        masked_n1, masked_n2 = n1, n2
        ranked = scipy.stats.rankdata(alldata, axis=1, nan_policy='omit')
    else:
        masked_n1 = np.count_nonzero(~np.isnan(x), axis=1)
        masked_n2 = np.count_nonzero(~np.isnan(y), axis=1)
        mask = alldata.copy()
        mask[mask == mask] = 1
        mask[mask != mask] = 0
        ranked = scipy.stats.rankdata(alldata, axis=1, nan_policy='omit') * mask
    x = ranked[:, :n1]
    s = np.nansum(x, axis=1)
    expected = masked_n1 * (masked_n1 + masked_n2 + 1) / 2.0
    statistic = (s - expected) / np.sqrt(masked_n1 *
                                         masked_n2 * (masked_n1 + masked_n2 + 1) / 12.0)
    pvalue = 2 * scipy.stats.distributions.norm.sf(abs(statistic))

    return statistic, pvalue


def kruskal_vectorized(a):
    n = np.asarray(list(map(lambda x: len(x[0]), a)))
    num_groups = len(a)

    alldata = np.concatenate(a, axis=1)
    ranked = scipy.stats.rankdata(alldata, axis=1, nan_policy='omit')
    if np.any(ranked.max(axis=1) == ranked.min(axis=1)):
        raise ValueError('All numbers are identical in kruskal')
    ties = tiecorrect_vectorized(ranked)

    j = np.insert(np.cumsum(n), 0, 0)
    ssbn = 0

    for i in range(num_groups):
        ssbn += square_of_sums(ranked[:, j[i]:j[i + 1]], axis=1) / n[i]

    totaln = np.sum(n, dtype=float)
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    df = num_groups - 1
    h /= ties

    return h, scipy.stats.distributions.chi2.sf([h], [df])[0]


def f_oneway_vectorized(*samples, axis=0):
    samples = [np.asarray(sample, dtype=float) for sample in samples]
    num_groups = len(samples)
    alldata = np.concatenate(samples, axis=axis)
    bign = np.sum(~np.isnan(alldata), axis=axis)
    # print('bign', bign)
    offset = np.nanmean(alldata, axis=axis, keepdims=True)
    alldata -= offset

    normalized_ss = square_of_sums(alldata, axis=axis) / bign
    # print('normalized_ss', normalized_ss)
    sstot = sum_of_squares(alldata, axis=axis) - normalized_ss
    ssbn = 0
    for sample in samples:
        ssbn += square_of_sums(sample - offset,
                                axis=axis) / np.sum(~np.isnan(sample), axis=axis)

    from scipy import special
    ssbn -= normalized_ss
    sswn = sstot - ssbn
    dfbn = num_groups - 1
    dfwn = bign - num_groups
    # print('dfwn', dfwn)
    msb = ssbn / dfbn
    msw = sswn / dfwn
    with np.errstate(divide='ignore', invalid='ignore'):
        f = msb / msw

    prob = special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf
    return f, prob


def core_corr_tensor(A, B, method='spearman'):
    if method == 'spearman':
        A = obtain_ranked(A)
        B = obtain_ranked(B)
        dof = np.count_nonzero(~np.isnan(A), axis=1) - 2
    elif method == 'pearson':
        dof = np.count_nonzero(~np.isnan(A), axis=1) / 2 -1

    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    corr = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))
    return np.unique(dof), corr


def core_corr(a, b, method='spearman'):
    if a.shape[0] == 1:
        a = np.repeat(a, b.shape[0], axis=0)
    mask_a = np.clip(a, 1, 1)
    mask_b = np.clip(b, 1, 1)
    a = a * mask_a * mask_b
    b = b * mask_a * mask_b
    if method == 'spearman':
        A = obtain_ranked(a)
        B = obtain_ranked(b)
        dof = np.count_nonzero(~np.isnan(a), axis=1) - 2
    elif method == 'pearson':
        A = a
        B = b
        dof = np.count_nonzero(~np.isnan(a), axis=1) / 2 -1
    else:
        raise ValueError('core_corr function only support spearman and pearson, please check your method parameter.')
    A_mA = A - np.nanmean(A,1)[:, None]
    B_mB = B - np.nanmean(B,1)[:, None]

    ssA = np.nansum(A_mA**2, axis=1)
    ssB = np.nansum(B_mB**2, axis=1)
    corr = np.nansum(A_mA * B_mB, 1)[None, :] / np.sqrt((ssA * ssB)[None, :])
    count = np.count_nonzero(~np.isnan(a), axis=1)
    return count, dof, corr


def pearson_pvalue(cor, ab):
    return 2 * scipy.special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(cor))))


def spearman_pvalue(cor, dof):
    t = cor * np.sqrt((dof / ((cor + 1.0) * (1.0 - cor))).clip(0))
    return 2 * scipy.stats.distributions.t.sf(np.abs(t), dof)


def statistic_func(statistic_values, statistic_method='ttest', ttest_kwargs={}):
    if statistic_method == 'ttest':
        statistic_value = scipy.stats.ttest_ind(*statistic_values,
                                                axis=1,
                                                equal_var=True,
                                                **ttest_kwargs)
    elif statistic_method == 'ranksums':
        statistic_value = ranksums_vectorized(statistic_values[0],
                                          statistic_values[1])

    return statistic_value