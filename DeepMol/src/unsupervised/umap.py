from unsupervised.baseUnsupervised import UnsupervisedLearn
from umap.parametric_umap import ParametricUMAP
import plotly.express as px

class UMAP(UnsupervisedLearn):
    """Class to perform Uniform Manifold Approximation and Projection (UMAP) .

        Wrapper around umap package.
        (https://github.com/lmcinnes/umap)
        """

    def __init__(self,
                 n_neighbors=15,
                 n_components=2,
                 metric='euclidean',
                 n_epochs=None,
                 learning_rate=1.0,
                 low_memory=True,
                 random_state=None):
        #TODO: comments
        """
        Parameters
        ----------
        n_neighbors: float (optional, default 15)
            The size of local neighborhood (in terms of number of neighboring
            sample points) used for manifold approximation. Larger values
            result in more global views of the manifold, while smaller
            values result in more local data being preserved. In general
            values should be in the range 2 to 100.
        """

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.low_memory = low_memory
        self.random_state = random_state


    def _runUnsupervised(self, plot=True):
        """Compute cluster centers and predict cluster index for each sample."""

        embedder = ParametricUMAP(n_neighbors=self.n_neighbors,
                                  n_components=self.n_components,
                                  metric=self.metric,
                                  n_epochs=self.n_epochs,
                                  learning_rate=self.learning_rate,
                                  low_memory=self.low_memory,
                                  random_state=self.random_state)

        embedding = embedder.fit_transform(self.features)

        if plot:
            if self.n_components != 2:
                raise ValueError('Only 2 components UMAP supported!')

            self._plot(embedding, self.dataset.y)
            #points(embedding, labels=self.dataset.y)

        return embedding

    def _plot(self, embedding, Y_train):

        print('2 Components UMAP: ')

        dic = {0: "Not Active (0)", 1: "Active (1)"}
        colors_map = []
        for elem in self.dataset.y:
            colors_map.append(dic[elem])

        fig = px.scatter(embedding, x=0, y=1,
                         color=colors_map, labels={'color': 'Class'},
                         title='UMAP:'
                         )
        fig.show()