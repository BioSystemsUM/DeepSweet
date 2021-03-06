from imblearn import over_sampling, under_sampling, combine
from sklearn.cluster import KMeans

from Datasets.Datasets import Dataset

#TODO: give artificial ids/mols to artificial molecules generated by these methods
class ImbalancedLearn(object):
    """Class for dealing with imbalanced datasets.

    A ImbalancedLearn sampler receives a Dataset object and performs over/under sampling.

    Subclasses need to implement a _sampling method to perform over/under sampling.
    """

    def __init__(self):
        if self.__class__ == ImbalancedLearn:
            raise Exception('Abstract class ImbalancedLearn should not be instantiated')

        self.features = None
        self.y = None

    def sample(self, train_dataset: Dataset):

        self.features = train_dataset.X

        self.y = train_dataset.y

        features, y = self._sample()

        train_dataset.X = features

        train_dataset.y = y

        return train_dataset


#########################################
# OVER-SAMPLING
#########################################

class RandomOverSampler(ImbalancedLearn):
    """Class to perform naive random over-sampling.

    Wrapper around ImbalancedLearn RandomOverSampler
    (https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html)

    Object to over-sample the minority class(es) by picking samples at random
    with replacement.
    """

    def __init__(self, sampling_strategy="auto", random_state=None):
        """
        Parameters
        ----------
        sampling_strategy: float, str, dict or callable, (default=’auto’)
            Sampling information to resample the data set.

            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.

            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.

            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.

            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.

        random_state int, RandomState instance or None, optional (default=None)
            Control the randomization of the algorithm.

            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def _sample(self):
        """Returns features resampled and y resampled."""
        ros = over_sampling.RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        return ros.fit_resample(self.features, self.y)


class SMOTE(ImbalancedLearn):
    """Class to perform Synthetic Minority Oversampling Technique (SMOTE) over-sampling.

    Wrapper around ImbalancedLearn SMOTE
    (https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.SMOTE.html)
    """

    def __init__(self,
                 sampling_strategy="auto",
                 random_state=None,
                 k_neighbors=5,
                 n_jobs=None):
        """
        Parameters
        ----------
        sampling_strategy: float, str, dict or callable, default=’auto’
            Sampling information to resample the data set.

            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.

            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.

            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.

            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.

        random_state: int, RandomState instance, default=None
            Control the randomization of the algorithm.

            If int, random_state is the seed used by the random number generator;

            If RandomState instance, random_state is the random number generator;

            If None, the random number generator is the RandomState instance used by np.random.

        k_neighbors: int or object, default=5
            If ``int``, number of nearest neighbours to used to construct synthetic
            samples.  If object, an estimator that inherits from
            :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
            find the k_neighbors.

        n_jobs: int, default=None
            Number of CPU cores used during the cross-validation loop. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _sample(self):
        """Returns features resampled and y resampled."""
        ros = over_sampling.SMOTE(sampling_strategy=self.sampling_strategy,
                                  random_state=self.random_state,
                                  k_neighbors=self.k_neighbors,
                                  n_jobs=self.n_jobs)
        return ros.fit_resample(self.features, self.y)

# TODO: add ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
#  for mixed data type such as continuous and categorical features add SMOTENC


#########################################
# UNDER-SAMPLING
#########################################

class ClusterCentroids(ImbalancedLearn):
    """Class to perform ClusterCentroids under-sampling.

    Wrapper around ImbalancedLearn ClusterCentroids
    (https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.ClusterCentroids.html)

    Perform under-sampling by generating centroids based on clustering.
    """

    def __init__(self,
                 sampling_strategy="auto",
                 random_state=None,
                 estimator=KMeans(),
                 voting='auto'):
        """
        Parameters
        ----------
        sampling_strategy: float, str, dict or callable, default=’auto’
            Sampling information to resample the data set.

            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.

            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.

            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.

            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.

        random_state: int, RandomState instance, default=None
            Control the randomization of the algorithm.

            If int, random_state is the seed used by the random number generator;

            If RandomState instance, random_state is the random number generator;

            If None, the random number generator is the RandomState instance used by np.random.

        k_neighbors: object, default=KMeans()
            Pass a sklearn.cluster.KMeans estimator.

        voting: str, default=’auto’
            Voting strategy to generate the new samples:

            If 'hard', the nearest-neighbors of the centroids found using the clustering algorithm will be used.

            If 'soft', the centroids found by the clustering algorithm will be used.

            If 'auto', if the input is sparse, it will default on 'hard' otherwise, 'soft' will be used.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.estimator = estimator
        self.voting = voting

    def _sample(self):
        """Returns features resampled and y resampled."""
        ros = under_sampling.ClusterCentroids(sampling_strategy=self.sampling_strategy,
                                              random_state=self.random_state,
                                              estimator=self.estimator,
                                              voting=self.voting)
        return ros.fit_resample(self.features, self.y)

class RandomUnderSampler(ImbalancedLearn):
    """Class to perform RandomUnderSampler under-sampling.

    Wrapper around ImbalancedLearn RandomUnderSampler
    (https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)

    Under-sample the majority class(es) by randomly picking samples with or without replacement.
    """

    def __init__(self,
                 sampling_strategy="auto",
                 random_state=None,
                 replacement=False):
        """
        Parameters
        ----------
        sampling_strategy: float, str, dict or callable, default=’auto’
            Sampling information to resample the data set.

            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.

            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.

            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.

            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.

        random_state: int, RandomState instance, default=None
            Control the randomization of the algorithm.

            If int, random_state is the seed used by the random number generator;

            If RandomState instance, random_state is the random number generator;

            If None, the random number generator is the RandomState instance used by np.random.

        replacement: bool, default=False
            Whether the sample is with or without replacement.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement

    def _sample(self):
        """Returns features resampled and y resampled."""
        ros = under_sampling.RandomUnderSampler(sampling_strategy=self.sampling_strategy,
                                                random_state=self.random_state,
                                                replacement=self.replacement)
        return ros.fit_resample(self.features, self.y)

# TODO: add CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN,
#  InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks


#########################################
# COMBINATION OF OVER AND UNDER-SAMPLING
#########################################

class SMOTEENN(ImbalancedLearn):
    """Class to perform SMOTEENN over and under-sampling.

    Wrapper around ImbalancedLearn SMOTEENN
    (https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTEENN.html)

    Over-sampling using SMOTE and cleaning using ENN.
    Combine over and under-sampling using SMOTE and Edited Nearest Neighbours.
    """

    def __init__(self,
                 sampling_strategy="auto",
                 random_state=None,
                 smote=None,
                 enn=None,
                 n_jobs=None):
        """
        Parameters
        ----------
        sampling_strategy: float, str, dict or callable, default=’auto’
            Sampling information to resample the data set.

            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.

            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.

            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.

            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.

        random_state: int, RandomState instance, default=None
            Control the randomization of the algorithm.

            If int, random_state is the seed used by the random number generator;

            If RandomState instance, random_state is the random number generator;

            If None, the random number generator is the RandomState instance used by np.random.

        smote: object, default=None
            The imblearn.over_sampling.SMOTE object to use. If not given, a imblearn.over_sampling.SMOTE object
            with default parameters will be given.

        enn: object, default=None
            The imblearn.under_sampling.EditedNearestNeighbours object to use. If not given, a
            imblearn.under_sampling.EditedNearestNeighbours object with sampling strategy=’all’ will be given.

        n_jobs: int, default=None
            Number of CPU cores used during the cross-validation loop. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.enn = enn
        self.n_jobs = n_jobs

    def _sample(self):
        """Returns features resampled and y resampled."""
        ros = combine.SMOTEENN(sampling_strategy=self.sampling_strategy,
                               random_state=self.random_state,
                               smote=self.smote,
                               enn=self.enn,
                               n_jobs=self.n_jobs)
        return ros.fit_resample(self.features, self.y)


class SMOTETomek(ImbalancedLearn):
    """Class to perform SMOTETomek over and under-sampling.

    Wrapper around ImbalancedLearn SMOTETomek
    (https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTETomek.html)

    Over-sampling using SMOTE and cleaning using Tomek links.
    Combine over- and under-sampling using SMOTE and Tomek links.
    """

    def __init__(self,
                 sampling_strategy="auto",
                 random_state=None,
                 smote=None,
                 tomek=None,
                 n_jobs=None):
        """
        Parameters
        ----------
        sampling_strategy: float, str, dict or callable, default=’auto’
            Sampling information to resample the data set.

            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.

            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.

            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.

            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.

        random_state: int, RandomState instance, default=None
            Control the randomization of the algorithm.

            If int, random_state is the seed used by the random number generator;

            If RandomState instance, random_state is the random number generator;

            If None, the random number generator is the RandomState instance used by np.random.

        smote: object, default=None
            The imblearn.over_sampling.SMOTE object to use. If not given, a imblearn.over_sampling.SMOTE object
            with default parameters will be given.

        tomek: object, default=None
            The imblearn.under_sampling.TomekLinks object to use. If not given, a imblearn.under_sampling.TomekLinks
            object with sampling strategy=’all’ will be given.

        n_jobs: int, default=None
            Number of CPU cores used during the cross-validation loop. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.tomek = tomek
        self.n_jobs = n_jobs

    def _sample(self):
        """Returns features resampled and y resampled."""
        ros = combine.SMOTETomek(sampling_strategy=self.sampling_strategy,
                                 random_state=self.random_state,
                                 smote=self.replacement,
                                 tomek=self.tomek,
                                 n_jobs=self.n_jobs)
        return ros.fit_resample(self.features, self.y)

# TODO: check the rest of imbalanced-learn