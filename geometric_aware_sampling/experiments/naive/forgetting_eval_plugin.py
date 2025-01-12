import torch
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class ForgettingEvalPlugin(SupervisedPlugin):
    def __init__(self):
        """
        Initializes the class centroids plugin with an empty dictionary to store centroids per class.
        """
        super().__init__()
        self.class_centroids = {}  # Dict stores centroids per class
        self.class_counts = {}  # Dict keeps per class for centroid calculation
        self.complete_dataset = []  # Complete dataset across experiences
        self.centroid_distances = []

    def compute_centroids(self, inputs, targets, model):
        # Reset centroids and counts
        with torch.no_grad():
            outputs = model.extract_last_layer(inputs)  # Compute model outputs

        class_centroids = {}
        class_counts = {}

        for label in torch.unique(targets):
            label = label.item()
            label_mask = targets == label
            label_outputs = outputs[label_mask].cpu()

            if label not in self.class_centroids:
                class_centroids[label] = torch.zeros_like(label_outputs[0])
                class_counts[label] = 0

            # Update centroid and count for the class
            class_centroids[label] += label_outputs.sum(dim=0)
            class_counts[label] += label_outputs.size(0)

        # Normalize centroids
        for label in class_centroids:
            class_centroids[label] /= class_counts[label]

        return class_centroids, class_counts

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        Hook executed after each training experience.
        Updates the centroids per class based on the entire dataset seen so far.
        """
        # Add the current experience's dataset to the complete dataset
        self.complete_dataset.extend(strategy.experience.dataset)

        # Process the entire dataset
        model = strategy.model.eval()  # Set model to evaluation mode

        inputs, targets = zip(*[(data[0], data[1]) for data in self.complete_dataset])

        inputs = torch.stack(list(inputs), dim=0).cuda()
        targets = torch.tensor(targets)

        self.class_centroids, self.class_counts = self.compute_centroids(
            inputs, targets, model
        )

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        Hook executed before training on the next experience.
        Computes and stores the Euclidean distance between centroids of old and new classes.
        """
        if not self.class_centroids:
            return

        current_experience_labels = set(strategy.experience.classes_in_this_experience)
        old_labels = set(self.class_centroids.keys()) - current_experience_labels
        new_labels = current_experience_labels

        # Compute new centroids for the current experience
        model = strategy.model.eval()  # Set model to evaluation mode
        current_dataset = strategy.experience.dataset
        inputs, targets = zip(*[(data[0], data[1]) for data in current_dataset])
        inputs = torch.stack(list(inputs), dim=0).cuda()
        targets = torch.tensor(targets)

        new_centroids, _ = self.compute_centroids(inputs, targets, model)

        distances = {}
        for old_label in old_labels:
            for new_label in new_labels:
                old_centroid = self.class_centroids[old_label]
                new_centroid = new_centroids[new_label]
                distances[(old_label, new_label)] = torch.dist(
                    old_centroid, new_centroid
                ).item()

        self.centroid_distances.append(distances)
        print(
            f"Centroid distances for experience {strategy.experience.current_experience}: {distances}"
        )
