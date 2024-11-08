import numpy as np
import starsim as ss
import numba as nb
import pandas as pd

class RandomNetwork(ss.Network):

    # meta = {
    #     'p1': ss.int_,
    #     'p2': ss.int_,
    #     'beta': ss.float_,
    # }

    def __init__(self, *, n_contacts: ss.Distribution, dynamic=True, layer_beta=1.0, **kwargs):
        """
        :param n_contacts: A distribution of contacts e.g., ss.delta(5), ss.neg_binomial(5,2)
        :param dynamic: If True, regenerate contacts each timestep
        """
        super().__init__(**kwargs)
        self.n_contacts = n_contacts
        self.dynamic = dynamic
        self.layer_beta = layer_beta

    def initialize(self, sim):
        super().initialize(sim)
        self.update(sim.people, force=True)

    @staticmethod
    @nb.njit
    def get_contacts(inds, number_of_contacts):
        """
        Efficiently generate contacts

        Note that because of the shuffling operation, each person is assigned 2N contacts
        (i.e. if a person has 5 contacts, they appear 5 times in the 'source' array and 5
        times in the 'target' array). Therefore, the `number_of_contacts` argument to this
        function should be HALF of the total contacts a person is expected to have, if both
        the source and target array outputs are used (e.g. for social contacts)

        adjusted_number_of_contacts = np.round(number_of_contacts / 2).astype(cvd.default_int)

        Whereas for asymmetric contacts (e.g. staff-public interactions) it might not be necessary

        Args:
            inds: List/array of person indices
            number_of_contacts: List/array the same length as `inds` with the number of unidirectional
            contacts to assign to each person. Therefore, a person will have on average TWICE this number
            of random contacts.

        Returns: Two arrays, for source and target


        """

        total_number_of_half_edges = np.sum(number_of_contacts)
        count = 0
        source = np.zeros((total_number_of_half_edges,), dtype=ss.int_)
        for i, person_id in enumerate(inds):
            n_contacts = number_of_contacts[i]
            source[count : count + n_contacts] = person_id
            count += n_contacts
        target = np.random.permutation(source)
        return source, target

    def update(self, people: ss.People, force: bool = True) -> None:
        """
        Regenerate contacts

        Args:
            force: If True, ignore the `self.dynamic` flag. This is required for initialization.

        """

        if not self.dynamic and not force:
            return

        number_of_contacts = self.n_contacts.sample(len(people))
        number_of_contacts = np.round(number_of_contacts / 2).astype(ss.int_)  # One-way contacts
        self.contacts.p1, self.contacts.p2 = self.get_contacts(people.uid.__array__(), number_of_contacts)
        self.contacts.beta = np.ones(len(self.contacts.p1), dtype=ss.float_) * self.layer_beta

class HouseholdNetwork(ss.Network):
    """
    Clustered household network with fixed, precomputed clusters

    """

    def __init__(self, clusters, *args, **kwargs):
        """

        Args:
            clusters: List of lists, ``[[person_ids],...]`` defining each cluster
            *args: Passed to ``VictoriaLayer``
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.clusters = clusters  # Store clusters for later use
        self.contacts.p1, self.contacts.p2 = self.cluster_arrays(clusters)
        self.contacts.beta = np.ones(len(self.contacts.p1), dtype=ss.float_)
        self.validate()

    @staticmethod
    def cluster_arrays(clusters):
        # Convert a list of lists of clusters into an edge representation
        # where each cluster is fully connected

        p1 = []
        p2 = []

        for cluster in clusters:
            for i, a in enumerate(cluster):
                for j, b in enumerate(cluster):
                    if j < i:
                        p1.append(a)
                        p2.append(b)

        return np.array(p1, dtype=ss.int_), np.array(p2, dtype=ss.int_)

## Fast choice implementation
# From https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
# by Jeremy Howard https://twitter.com/jeremyphoward/status/955136770806444032
@nb.njit
def _sample(n, q, J, r1, r2):
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i] * lj))
        if r2[i] < q[kk]:
            res[i] = kk
        else:
            res[i] = J[kk]
    return res

def parse_age_range(x):
    if "+" in x:  # Handle "95+"
        age_lower = float(x.split("+")[0])
        age_upper = np.inf
    elif "-" in x:  # Handle "5-9"
        age_lower = float(x.split("-")[0])
        age_upper = float(x.split("-")[1])
    else:  # Handle "5 to 9"
        age_lower = float(x.split("to")[0])
        age_upper = float(x.split("to")[1])
    return age_lower, age_upper

class AliasSample:
    def __init__(self, probs):
        self.K = K = len(probs)
        self.q = q = np.zeros(K)
        self.J = J = np.zeros(K, dtype=ss.int_)

        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw_one(self):
        K, q, J = self.K, self.q, self.J
        kk = int(np.floor(np.random.rand() * len(J)))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def draw_n(self, n):
        r1, r2 = np.random.rand(n), np.random.rand(n)
        return _sample(n, self.q, self.J, r1, r2)

class MatrixSampler:
    # Wraps the fast routines for sample and AliasSample for efficient sampling

    def __init__(self, df):

        # To sample contacts for a reference age, select the row which that person's age corresponds to
        # We therefore need to normalize so that each row sums to 1
        df = df.div(df.sum(axis=1), axis=0)

        # Precompute samplers for each reference age bin
        self._samplers = [AliasSample(df.iloc[i, :].values) for i in range(df.shape[0])]

        # Store bins
        self._bin_lower = df.index.map(lambda x: parse_age_range(x)[0])
        self._bin_upper = df.index.map(lambda x: parse_age_range(x)[1])

    def sample_bins(self, reference_age: int, n: int) -> np.ndarray:
        """
        Sample bins from age matrix

        Returns sampled column indices from mixing matrix based on distribution defined
        by row corresponding to reference person

        Args:
            reference_age: Age of reference person
            n: Number of bins to return

        Returns: Array of bin indices

        """
        idx = np.digitize(reference_age, self._bin_lower) - 1  # First, find the index of the bin that the reference person belongs to
        sampled_bins = self._samplers[idx].draw_n(n)
        return sampled_bins

    def sample_ages(self, reference_age: int, n: int) -> np.ndarray:
        """
        Sample a cluster of ages

        For example, a household. `n` is the number of people in the cluster. Note that
        if `n=1` then the returned array will only contain the reference person's age.

        Args:
            reference_age: Age of reference person
            n: Number of people in the cluster

        Returns: Array of length (n) with household ages

        """
        # Populate a sample of length `n` based on a reference age
        # The output array includes the reference person
        ages = [reference_age]  # The reference person is in the household/location
        if n > 1:
            sampled_bins = self.sample_bins(reference_age, n - 1)
            for bin in sampled_bins:
                ages.append(int(round(np.random.uniform(self._bin_lower[bin] - 0.5, self._bin_upper[bin] + 0.5))))
        return np.array(ages)

def _make_households(n_households, pop_size, household_heads, mixing_matrix):
    """

    The mixing matrix is a direct read of the CSV file, with index corresponding to 'Age group' i.e.

    >>> mixing_matrix = pd.read_csv('mixing_H.csv',index_col='Age group')
    >>> mixing_matrix
                   0 to 4    5 to 9
        Age group
        0 to 4     0.659868  0.503965
        5 to 9     0.314777  0.895460


    :param n_households:
    :param pop_size:
    :param household_heads:
    :return:
        h_clusters: a list of lists in which each sublist contains
                    the IDs of the people who live in a specific household
        ages: flattened array of ages, corresponding to the UID positions
    """

    ms = MatrixSampler(mixing_matrix)

    h_clusters = []
    uids = np.arange(0, pop_size)
    ages = np.zeros(pop_size, dtype=ss.float_)
    h_added = 0
    p_added = 0

    for h_size, h_num in n_households.items():
        for household in range(h_num):
            head = household_heads[h_added]
            household_ages = ms.sample_ages(head, h_size)
            # add ages to ages array
            ub = p_added + h_size
            ages[p_added:ub] = household_ages
            # get associated UID that defines a household cluster
            h_ids = uids[p_added:ub]
            h_clusters.append(h_ids)
            # increment sliding windows
            h_added += 1
            p_added += h_size
    return h_clusters, ages

def generate_household_clusters(n_people: int, mixing: pd.DataFrame, reference_ages: pd.Series, households: pd.Series):
    # First work out how many households of each size we need
    total_people = sum(households.index * households.values)  # total_people = household_size * n_households
    household_percent = households / total_people
    n_households = (n_people * household_percent).round().astype(int)
    n_households[1] += n_people - sum( n_households * n_households.index)  # adjust single-person households to fill the gap

    # Then, select a reference person age for the first person in each household
    household_heads = np.random.choice(reference_ages.index, size=sum(n_households), p=reference_ages.values / sum(reference_ages))

    h_clusters, ages = _make_households(n_households, n_people, household_heads, mixing)

    return h_clusters, ages
