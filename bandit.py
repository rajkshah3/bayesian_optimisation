from scipy.stats import bernoulli

class bandit:
    def __init__(self,winning_fraction):
        self.winning_fraction = winning_fraction

    def pull(self) -> bool:
        return bernoulli.rvs(self.winning_fraction)


if __name__ == '__main__':
    b = bandit(winning_fraction=0.4)
    for i in range(10):
        print(b.pull())
