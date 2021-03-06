import numpy as np
import math
from scipy.stats import nakagami
from scipy.special import gammainc


class UserChannel:
    def __init__(self, pid, m, ms, db, d, n, nd, pj, dbi, noise, alpha, a, b, size):
        self.pid = pid
        self.m = m  # multipath
        self.ms = ms  # shadowing
        self.db = db  # db of fading
        self.d = d  # distance
        self.alpha = alpha  # large scale
        self.a = a  # modulation
        self.b = b  # modulation
        self.n = n  # number of antennas
        self.nd = nd  # paths of interference
        self.pj = pj  # power of interference (in each path)
        self.dbi = dbi  # db of interference
        self.noise = noise  # noise
        self.size = size

        np.random.seed(0)
        self.F = self._fisher()
        self.I = self._inter()

    def _fisher(self):
        sigma = 1.
        temp = 1
        SNR_mean = pow(10, self.db / 10)
        Y = np.zeros(self.size)
        Y = np.array(Y)
        while temp <= self.n:
            Xn = nakagami.rvs(self.m, loc=0, scale=np.sqrt(sigma), size=self.size)
            A = 1 / nakagami.rvs(self.ms, loc=0, scale=np.sqrt(sigma), size=self.size)
            R2 = pow(A, 2) * pow(Xn, 2)
            T = SNR_mean * R2 / np.mean(R2)
            Y = [T[i] + Y[i] for i in range(len(T))]
            temp += 1
        return Y

    def _inter(self):
        sigma = 1.
        SNR_mean = pow(10, self.dbi / 10)
        Xn = nakagami.rvs(self.nd, loc=0, scale=np.sqrt(sigma), size=self.size)
        R2 = pow(Xn, 2)
        Yi = SNR_mean * R2 / np.mean(R2)
        Yi = Yi * self.pj
        return Yi

    def _nchoosek(self, n, k):
        res = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        return float(res)

    def get_packet_drop_prob(self, total_power, priority):
        power_per_packet = [total_power * prio_ / sum(priority)
                            for prio_ in priority]

        BEP = []
        for power in power_per_packet:
            z = [power * pow(self.d, -self.alpha) * self.F[i] / (self.noise + self.I[i]) for i in range(len(self.F))]
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
    def __init__(self, num_persons, Ms=None, MSs=None, Ds=None, PIs=None):
        Ms = Ms if Ms else [2, 2, 5, 5, 2, 2, 5]
        MSs = MSs if MSs else [2, 4, 2, 2, 4, 4, 4]
        DBs = [-3, -3, -3, -3, -5, -5, -3]
        Ds = Ds if Ds else [10, 10, 10, 12, 10, 10, 12]
        Ns = [3, 3, 3, 3, 3, 4, 3]
        NDs = [2, 2, 2, 2, 2, 2, 4]
        PIs = PIs if PIs else [2, 2, 2, 2, 2, 2, 2]
        DBIs = [-3, -3, -3, -3, -3, -3, -3]
        self.user_channels = [UserChannel(pid, Ms[pid], MSs[pid], DBs[pid], Ds[pid],
                                          Ns[pid], NDs[pid], PIs[pid], DBIs[pid],
                                          noise=1, alpha=2, a=1, b=0.5, size=10000)
                              for pid in range(num_persons)]

    def get_user_channel(self, pid):
        return self.user_channels[pid]


if __name__ == "__main__":
    num_persons = 7
    fading_channel = FadingChannel(num_persons)
    for pid in range(num_persons):
        user_channel = fading_channel.get_user_channel(pid)
        drop_prob = user_channel.get_packet_drop_prob(
            1000, [1, 1, 1, 1, 1])
        print(drop_prob)
