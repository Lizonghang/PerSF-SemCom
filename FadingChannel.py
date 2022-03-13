import numpy as np
import math
from scipy.stats import nakagami
from scipy.special import gammainc


class UserChannel:
    def __init__(self, pid, m, ms, db, d, alpha, a, b, size):
        self.pid = pid
        self.m = m
        self.ms = ms
        self.db = db
        self.d = d
        self.alpha = alpha
        self.a = a
        self.b = b
        self.size = size

        np.random.seed(0)
        self.F = self._fisher()

    def _fisher(self):
        sigma = 1.
        Xn = nakagami.rvs(self.m, loc=0, scale=np.sqrt(sigma), size=self.size)
        A = 1 / nakagami.rvs(self.ms, loc=0, scale=np.sqrt(sigma), size=self.size)
        R2 = pow(A, 2) * pow(Xn, 2)
        SNR_mean = pow(10, self.db / 10)
        Y = SNR_mean * R2 / np.mean(R2)
        return Y

    def _nchoosek(self, n, k):
        res = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        return float(res)

    def get_packet_drop_prob(self, total_power, priority):
        power_per_packet = [total_power * prio_ / sum(priority)
                            for prio_ in priority]
        BEP = []
        for power in power_per_packet:
            z = power * pow(self.d, -self.alpha) * self.F
            z2 = (math.gamma(self.b) * (1 - gammainc(self.b, self.a * z))) \
                 / (2 * math.gamma(self.b))
            BEP.append(np.mean(z2))

        PL = 20  # packet length is PL bits
        E = 3  # the maximum error correcting capability
        PDP = []

        for bep_ in BEP:
            PDPtemp = 0
            BEPf = 1 - bep_
            for it in range(0, E):
                PDPtemp += self._nchoosek(PL, it) * pow(bep_, it) * pow(BEPf, PL - it)
            PDP.append(1 - PDPtemp)

        return PDP


class FadingChannel:
    def __init__(self, num_persons):
        Ms = [1.5, 1.5, 6, 6, 1.5, 1.5, 6]
        MSs = [2, 2, 2, 2, 8, 8, 8]
        DBs = [-10, -15, -10, -10, -10, -15, -10]
        Ds = [10, 10, 10, 16, 10, 10, 16]

        self.user_channels = [UserChannel(pid, Ms[pid], MSs[pid], DBs[pid], Ds[pid],
                                          alpha=2, a=1, b=0.5, size=10000)
                              for pid in range(num_persons)]

    def get_user_channel(self, pid):
        return self.user_channels[pid]


if __name__ == "__main__":
    num_persons = 7
    fading_channel = FadingChannel(num_persons)
    for pid in range(num_persons):
        user_channel = fading_channel.get_user_channel(pid)
        drop_prob = user_channel.get_packet_drop_prob(
            8000, [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7])
        print(drop_prob)
