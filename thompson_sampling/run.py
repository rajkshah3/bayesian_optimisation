"""
todo: Optimise sampling strategy of bandit posterior
todo: Calculate maximum probability across all win_probs for score and pull count

Current sampling strategy:
    MC accept-reject sampling of distribution, max_prob=1
"""

import numpy
from scipy.stats import binom

class ThomsonSampler:

    def __init__(self,number_of_bandits,prior_range=[0,1]):
        self.number_of_bandits = number_of_bandits
        self.bandit_scores = {b:0 for b in range(self.number_of_bandits)}
        self.bandit_pull_counts = {b:0 for b in range(self.number_of_bandits)}

        self.bandits = [ThomsonBandit(prior_range) for b in range(self.number_of_bandits)]

    def pick_move(self):
        bandit_probs = [b.get_win_prob() for b in self.bandits]
        best_bandit = max(range(len(bandit_probs)), key=bandit_probs.__getitem__)
        return best_bandit

    def update_bandit(self,bandit_number,score):
        self.bandits[bandit_number].update(score)

class ThomsonBandit:
    def __init__(self,prior_range=[0,1],prior='flat',distribution_samples=20):
        self.score = 0
        self.pull_count = 0
        self.prior_range = prior_range
        self.recalculate_win_prob = True
        self.distribution_samples = distribution_samples
        if prior == 'flat':
            self.prior_prob = 1
        else:
            print(f"Prior of type {prior} has not been implemented!")
            exit()

        self.max_probability = 1

    def get_prior(self,*args,**kwargs):
        return self.prior_prob

    def update(self,score):
        self.recalculate_win_prob = True
        self.update_score(score)
        self.update_pull_count()
        self.update_max_probability()
        # print(f"Maximum probability: {self.max_probability}")
        # self.test_update_max_probability()

    def test_update_max_probability(self):

        import numpy as np
        small_number = 0.00001
        win_ratio = self.get_peak_binom()
        upper_bound = min([1,win_ratio + small_number])
        lower_bound = max([0,win_ratio -  small_number])

        upper_bound_probability = self.calc_biomial_probability(upper_bound)
        lower_bound_probability = self.calc_biomial_probability(lower_bound)

        assert self.max_probability >= upper_bound_probability
        assert self.max_probability >= lower_bound_probability

        random_test_cases = np.random.random(10)
        for case in random_test_cases:
            probability = self.calc_biomial_probability(case)
            assert self.max_probability >= probability


    def get_peak_binom(self):
        return self.score/self.pull_count

    def update_max_probability(self):
        win_ratio = self.get_peak_binom()
        self.max_probability = self.calc_biomial_probability(win_ratio)

    def update_score(self,score):
        self.score += score

    def update_pull_count(self):
        self.pull_count += 1

    def get_random_comparison(self):
        return numpy.random.random() * self.max_probability

    def get_win_prob(self):
        if(self.recalculate_win_prob):
            self.win_prob = self.calc_win_prob()
            self.recalculate_win_prob = False
        return self.win_prob

    def calc_win_prob(self):
        probabilities = []
        for _ in range(self.distribution_samples):
            while True:
                random_prob = self.random_sample()
                prior_scale = self.get_prior(random_prob)
                likelihood = self.calc_biomial_probability(random_prob)
                random_number = self.get_random_comparison()*self.max_probability

                if(random_number < likelihood*prior_scale):
                    probabilities.append(random_prob)
                    break

        return sum(probabilities)/len(probabilities)

    def calc_biomial_probability(self,win_prob):
        if(self.pull_count == 0):
            return 1
        return binom.pmf(self.score,self.pull_count,win_prob)

    def random_sample(self):
        random_number = numpy.random.random()
        interval = self.prior_range[1] - self.prior_range[0]
        return interval*random_number + self.prior_range[0]

def test_thompson_sampler(bandits, turns):
    wins = 0
    sampler = ThomsonSampler(number_of_bandits=len(bandits))
    pull_counter = {b: 0 for b, _ in enumerate(bandits)}

    for t in range(turns):
        bandit_to_pull = sampler.pick_move()
        pull_counter[bandit_to_pull] = pull_counter[bandit_to_pull] + 1
        score = bandits[bandit_to_pull].pull()
        sampler.update_bandit(bandit_number=bandit_to_pull, score=score)
        wins += score

    return pull_counter, wins

if __name__ == '__main__':
    from bandit import bandit
    import numpy as np
    bandits = [bandit(prob) for prob in np.random.random(4)]
    for i in range(100):
        test_thompson_sampler(bandits,turns=100)
