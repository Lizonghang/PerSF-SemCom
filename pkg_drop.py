import numpy as np
import math
from scipy.stats import nakagami
from scipy.special import gammainc


def Fisher(m, ms, SNR_mean_dB, Num):
    Sigma = 1
    Xn = nakagami.rvs(m, loc=0, scale=np.sqrt(Sigma), size=Num)
    A = 1 / nakagami.rvs(ms, loc=0, scale=np.sqrt(Sigma), size=Num)
    R2 = pow(A, 2) * pow(Xn, 2)
    SNR_mean = pow(10, SNR_mean_dB / 10)
    Y = SNR_mean * R2 / np.mean(R2)
    return Y


def nchoosek(n, k):
    res = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    return float(res)


def PDPget(PZ, Weight):
    np.random.seed(0)

    Size = 10000
    P = []
    for i in Weight:
        P.append(PZ * i / sum(Weight))

    m = 5
    ms = 5
    db = -10
    f = Fisher(m, ms, db, Size)

    d = 10
    alpha = 2

    a = 1
    b = 0.5

    BEP = []
    for i in P:
        z = i * pow(d, -alpha) * f
        z2 = (math.gamma(b) * (1 - gammainc(b, a * z))) / (2 * math.gamma(b))
        BEP.append(np.mean(z2))

    PL = 20  # packet length is PL bits
    E = 3  # the maximum error correcting capability、
    PDP = []
    it = 0
    PDPtemp = 0
    for bep in BEP:
        BEPf = 1 - bep
        while it < E:
            PDPtemp = PDPtemp + nchoosek(PL, it) * pow(bep, it) * pow(BEPf, PL - it)
            it = it + 1
        PDP.append(1 - PDPtemp)
        PDPtemp = 0
        it = 0
    return PDP


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(PDPget(800, [0.02, 0.04, 0.5]))
