from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from avalanche.training.plugins import (
    SupervisedPlugin,
)
from avalanche.training.templates import SupervisedTemplate
from sklearn.decomposition import PCA


class RepresentationPlugin(SupervisedPlugin):
    """
    Plugin to compute the representation of the data in the last layer of the model.

    We use `after_training_exp` since we are interested in the representation of the
    classes after training on a task / certain classes = an experience

    Since the number of dimensions in this representation is too high, we project
    them to two or three dimensions using PCA (fitted on the data seen so far).
    We plot all the data learnt so far (previous and current experiences) as well as the next experience

    :param n_components: the number of dimensions in the space into which
        the PCA should project the last layer, two and three are supported values
    :param dl_batch_size: the number of samples per batch in the dataloader
        used for loading the samples from the experiences
    :param rdm_sz: Since many points in a visualisation overlap, if this parameter is set,
        only a random subset of the data is plotted. Set to None if there should be no sampling
    """

    def __init__(
        self, n_components: int = 2, dl_batch_size: int = 32, rdm_sz: int = 2000
    ):
        super().__init__()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.dl_batch_size = dl_batch_size
        self.rdm_sz = rdm_sz

    def plot_data(
        self,
        data,
        ys_eids,
        strategy: "SupervisedTemplate",
        writer: SummaryWriter = None,
        random_n: int = None,
    ):
        """
        Plots 2D or 3D data depending on the dimensionality of the input. Supports coloring by class and
        different markers by experience ID (second column of ys_eids).
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        if ys_eids.shape[1] != 2:
            raise ValueError("Second argument must include a class and an eid column")
        if random_n:
            rdm_selection = np.random.choice(data.shape[0], random_n, replace=False)
            data = data[rdm_selection]
            ys_eids = ys_eids[rdm_selection]
            sample_text = "a random sample of size " + str(random_n)
        else:
            sample_text = "the full sample"
        colors = plt.cm.viridis(
            np.linspace(
                0,
                1,
                len(strategy.experience.classes_seen_so_far)
                + (
                    strategy.experience.benchmark.n_classes_per_exp[
                        strategy.experience.current_experience + 1
                    ]
                    if strategy.experience.current_experience + 1
                    < len(strategy.experience.benchmark.n_classes_per_exp)
                    else 0
                ),
            )
        )
        # to access the colours in colors
        classes_to_idx = {c: i for i, c in enumerate(np.unique(ys_eids[:, 0]))}
        if data.shape[1] == 2:
            # 2D Scatter Plot
            fig = plt.figure()
            ax = fig.add_subplot()
            for i, point in enumerate(data):
                c = colors[classes_to_idx[ys_eids[i, 0]]]
                marker = (
                    "o"
                    if ys_eids[i, 1] <= strategy.experience.current_experience
                    else "x"
                )
                ax.scatter(
                    point[0],
                    point[1],
                    color=c,
                    marker=marker,
                    alpha=0.7,
                    s=100,
                )
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(
                "o: learned classes, x: future classes, colours: different classes"
            )
            if writer:
                writer.add_figure(
                    f"PCA of the last layer in experience {strategy.experience.current_experience} using "
                    + sample_text,
                    fig,
                )

        elif data.shape[1] == 3:
            # 3D Scatter Plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection="3d")
            for i, point in enumerate(data):
                c = colors[classes_to_idx[ys_eids[i, 0]]]
                marker = (
                    "o"
                    if ys_eids[i, 1] <= strategy.experience.current_experience
                    else "x"
                )
                ax.scatter(
                    point[0],
                    point[1],
                    point[2],
                    color=c,
                    marker=marker,
                    alpha=0.7,
                    s=100,
                )
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(
                "o: learned classes, x: future classes, colours: different classes"
            )
            if writer:
                writer.add_figure(
                    f"PCA of the last layer in experience {strategy.experience.current_experience} using "
                    + sample_text,
                    fig,
                )

        else:
            raise ValueError("Data must have 2 or 3 features (columns).")

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        # need to know the number of samples per experience in order to allocate
        # an array to save our results
        rows_per_exp = {
            exp.current_experience: len(exp.dataset)
            for exp in strategy.experience.benchmark.train_stream
        }
        # learned samples, i.e., past and current experience
        learned_rows = sum(
            v
            for k, v in rows_per_exp.items()
            if k <= strategy.experience.current_experience
        )
        # rows "of interest": past + current + one future experience
        num_rows = learned_rows + rows_per_exp.get(
            strategy.experience.current_experience + 1, 0
        )
        representation = np.zeros((num_rows, strategy.model.linear.in_features))
        ys_eids = np.zeros((num_rows, 2))
        # used for accessing the numpy arrays
        row_idx = 0
        for exp in strategy.experience.benchmark.train_stream:
            # only want to include data in the representation of the data learnt so far
            # and the next task
            if exp.current_experience <= strategy.experience.current_experience + 1:
                data_loader = DataLoader(
                    exp.dataset, batch_size=self.dl_batch_size, shuffle=False
                )
                for x, y, eid in data_loader:
                    # third value: experience ID
                    x = x.to(strategy.device)

                    last_layer = (
                        strategy.model.extract_last_layer(x).detach().to("cpu").numpy()
                    )
                    representation[
                        row_idx : (
                            row_idx + min(self.dl_batch_size, last_layer.shape[0])
                        )
                    ] = last_layer
                    ys_eids[
                        row_idx : row_idx + min(self.dl_batch_size, y.shape[0]), 0
                    ] = y
                    ys_eids[
                        row_idx : row_idx + min(self.dl_batch_size, eid.shape[0]), 1
                    ] = eid
                    row_idx += min(self.dl_batch_size, eid.shape[0])
        if ~representation.any(axis=0).any():
            print(
                "WARNING: representation extracted from the last layer contains zero rows"
            )
        # fit PCA on the data already seen (learned_rows) but project all rows
        # (including the future one)
        projected_repr = self.pca.fit(representation[:learned_rows]).transform(
            representation
        )
        print(
            f"PCA representation explained variance: {round(self.pca.explained_variance_ratio_.sum() * 100, 1)}%"
        )
        self.plot_data(
            projected_repr,
            ys_eids,
            strategy,
            writer=kwargs["tensorboard_logger"].writer,
            random_n=self.rdm_sz,
        )
