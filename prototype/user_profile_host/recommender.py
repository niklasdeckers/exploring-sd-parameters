import numpy as np
import random
import torch
import warnings
from abc import abstractmethod, ABC
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition import qUpperConfidenceBound
from botorch.exceptions import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from .utils import get_unnormalized_value

warnings.simplefilter("ignore", category=InputDataWarning)


class Recommender(ABC):  # ABC = Abstract Base Class
    """
    A Recommender class instance derives recommended samples for the next iteration.
    In other words:
    Multiple alterations of the current user profile (first iteration: random) are returned.
    It is important to note, that the user profile is a vector in the user space,
    i.e. a low dimensional subspace of the CLIP space.
    Hence, one assumes that points generated in the user space which are projected back to the CLIP space
    correspond to valid embeddings, i.e. produce meaningful images.
    The method used for generation depends on the user choice.
    It is possible to generate embeddings via the following methods:
    1. Single point generation with weighted axes:
        Some axes spanning the user space may convey more information than others.
        Hence, axes should be weighted differently according to their influence.
        There are two implementations: One where the points are on the surface of a sphere and one where they are not.
    2. Random generation:
        Random points in the user space are generated.
    3. Function-based generation:
        In this scenario, one doesn't want to optimize the position of the user profile (a point) in the suer profile
        space and use this position to generate new generations, but one chooses multiple points in fascinating
        regions of the user sspace and requests feedback of the user to "learn" the space.
        The choice of the points is based on an acquisition function, e.g. a Gaussian process.
    4. Dirichlet generation:
        A Dirichlet distribution is used to generate points in the user space.
        The concentration parameter is multiplied by the user profile to influence the distribution.
        The method increases the degree of exploitation with each recommendation.
        The Dirichlet distribution is chosen because it generates points on a simpley which produce roughly uniformly
        distributed samples in the high dimensional CLIP space after transformation & norming them.
    """

    @abstractmethod
    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        """
        :param user_profile: Encodes the user profile in the low-dimensional user profile space. Randomly initialized.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :param beta: Trade-off between exploration and exploitation. Higher beta means more exploitation.
            Must be in [0, 1].
        :return: A tensor of recommendations, i.e. n_recommendations many low-dimensional embeddings.
        """
        pass


class BaselineRecommender(Recommender):
    # SDXL MIGRATION: no change needed — height/width/in_channels are already constructor params
    # rather than hardcoded, so this picks up SDXL's 1024x1024 (and unchanged 4-channel latents)
    # automatically via whatever UserProfileHost passes in. Note this recommender only produces
    # latents, not embeddings, so it has no bearing on the new pooled-embedding handling
    # (see UserProfileHost.inv_transform's BASELINE branch for that).

    def __init__(self, n_latent_axis, in_channels, height, width, init_noise_sigma, seed: int = 42):
        self.n_latent_axis = n_latent_axis
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.init_noise_sigma = init_noise_sigma
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        """
        :param user_profile: A point in the low-dimensional user profile space.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :param beta: Not used in this recommender.
        :return: Tensor of shape (n_recommendations, n_dims) containing the samples on surface of sphere with center
            user_profile where n_dims is the dimensionality of the user_profile.
        """
        # Return random recommendations
        random_latents = torch.randn((n_recommendations, self.in_channels, self.height // 8, self.width // 8),
                                     generator=self.generator) * self.init_noise_sigma
        return random_latents


class SimpleRandomRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        """
        :param user_profile: A point in the low-dimensional user profile space.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :param beta: Not used in this recommender.
        :return: Tensor of shape (n_recommendations, n_dims) containing the samples on surface of sphere with center
            user_profile where n_dims is the dimensionality of the user_profile.
        """
        return None  # Placeholder


class RandomRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis, seed: int = 42):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.generator = random.Random(seed)

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        """
        :param user_profile: A point in the low-dimensional user profile space.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :param beta: Not used in this recommender.
        :return: Tensor of shape (n_recommendations, n_dims) containing the samples on surface of sphere with center
            user_profile where n_dims is the dimensionality of the user_profile.
        """
        # Return random recommendations
        alpha = torch.ones(self.n_axis)
        torch.manual_seed(
            self.generator.randint(0, 1000000))  # global seed, bc dirichlet doesn't support generator parameter
        dist = torch.distributions.dirichlet.Dirichlet(alpha)
        random_user_embeddings = dist.sample(sample_shape=(n_recommendations,))
        return random_user_embeddings


class SinglePointWeightedAxesRecommender(Recommender):

    def __init__(self, n_embedding_axis: int, n_latent_axis: int, seed: int = 42, ):
        """
        :param n_embedding_axis: Number of axes in the embedding space.
        :param n_latent_axis: Number of axes in the latent space.
        :param latent_bounds: Bounds for the latent space.
        :param embedding_bounds: Bounds for the embedding space. The lower bound should not be smaller than 0, because
            experimental evaluation led to the findings that negative bounds produce noisy generated images.
        """
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.bounds = (0., 1.)
        self.generator = random.Random(seed)

        # Define bounds for search space
        self.bounds = torch.tensor([
            # lower bounds (1, n_axis)
            [self.bounds[0] for i in range(self.n_embedding_axis)] + [self.bounds[0] for i in
                                                                      range(self.n_latent_axis)],
            # upper bounds (1, n_axis)
            [self.bounds[1] for i in range(self.n_embedding_axis)] + [self.bounds[1] for i in
                                                                      range(self.n_latent_axis)]
        ])

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = 0) -> Tensor:
        """
        Recommends embeddings based on the user profile, axes of the user space and the number of recommendations.
        Random weights are used to interpolate between the user profile and the axes.
        :param user_profile: Low-dimensional user profile.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation.
            Must be in [0, 1]. 0 means full exploration, 1 means full exploitation.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        axes = torch.eye(user_profile.shape[0])

        # distance of embedding bounds to user profile to find range to sample from
        lower_sampling_ranges = self.bounds[0] - user_profile
        upper_sampling_ranges = self.bounds[1] - user_profile

        alpha = torch.ones(self.n_axis)  # Concentration parameter (uniform)
        torch.manual_seed(
            self.generator.randint(0, 1000000))  # global seed, bc dirichlet doesn't support generator parameter
        distribution = torch.distributions.dirichlet.Dirichlet(alpha)
        weights_dirichlet = distribution.sample(sample_shape=(n_recommendations,))

        # scale to bounds to ranges & scale with exploration factor
        weights = ((1 - beta) * (
                weights_dirichlet * (upper_sampling_ranges - lower_sampling_ranges) + lower_sampling_ranges))

        # interpolate between user profile and axes, user user_profile as reference point
        return user_profile + weights @ axes


class DirichletRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis, seed: int = 42):
        """
        Initializes the Dirichlet Recommender.
        :param n_embedding_axis: Number of embedding axes (i.e. derived from prompt).
        :param n_latent_axis: Number of latent axes (i.e. 'noise').
        """
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.generator = random.Random(seed)

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = 0) -> Tensor:
        """
        Recommends embeddings based on the user profile, the number of recommendations and the trade-off between
        exploration and exploitation.
        :param user_profile: Low-dimensional user profile containing embeddings and preferences.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation. Higher beta means more exploitation.
            Must be in [0, 1]. 0 means full exploration, 1 means full exploitation.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        beta = get_unnormalized_value(beta, 1, 250)
        alpha = ((torch.ones(self.n_axis) * user_profile).reshape(-1) * beta)
        torch.manual_seed(
            self.generator.randint(0, 1000000))  # global seed, bc dirichlet doesn't support generator parameter
        dist = torch.distributions.dirichlet.Dirichlet(alpha)
        search_space = dist.sample(sample_shape=(n_recommendations,))

        return search_space


class DiverseDirichletRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis, seed: int = 42):
        """
        Initializes the Dirichlet Recommender.
        :param n_embedding_axis: Number of embedding axes (i.e. derived from prompt).
        :param n_latent_axis: Number of latent axes (i.e. 'noise').
        """
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.generator = random.Random(seed)
        self.np_generator = np.random.default_rng(seed)

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = 0) -> Tensor:
        """
        Recommends embeddings based on the user profile, the number of recommendations and the trade-off between
        exploration and exploitation.
        :param user_profile: Low-dimensional user profile containing embeddings and preferences.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation. Higher beta means more exploitation.
            Must be in [0, 1]. 0 means full exploration, 1 means full exploitation.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        beta = get_unnormalized_value(beta, 1, 500)
        user_embeddings, preferences = user_profile
        # Change preferences to numpy
        preferences = preferences.numpy()

        new_recommendations = []
        for i_rec in range(n_recommendations):
            # Draw a random embedding from previously iterations weighted by user preference
            idx = self.np_generator.choice(range(preferences.shape[0]), p=preferences / np.sum(preferences))

            # Select the respective user_embedding as a center
            center = user_embeddings[idx]

            # Build a dirichlet distribution around it
            alpha = ((torch.ones(self.n_axis) * center).reshape(-1) * beta)
            torch.manual_seed(
                self.generator.randint(0, 1000000))  # global seed, bc dirichlet doesn't support generator parameter
            dist = torch.distributions.dirichlet.Dirichlet(alpha)

            # Sample one sample
            sample = dist.sample(sample_shape=(1,))
            new_recommendations.append(sample)

        new_recommendations = torch.cat(new_recommendations)
        return new_recommendations


class BayesianRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis, n_points_per_axis: int = 3, seed: int = 42):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.n_points_per_axis = n_points_per_axis
        self.bounds = [0., 1.]
        self.generator = random.Random(seed)

    def build_search_space(self):
        n_samples = min(max(self.n_axis * 5 ** (self.n_axis // 2), 1000), 200000)
        alpha = torch.ones(self.n_axis)
        torch.manual_seed(
            self.generator.randint(0, 1000000))  # global seed, bc dirichlet doesn't support generator parameter
        dist = torch.distributions.dirichlet.Dirichlet(alpha)
        search_space = dist.sample(sample_shape=(n_samples,))
        return search_space

    def recommend_embeddings(self, user_profile: Tensor = None, n_recommendations: int = 5,
                             beta: float = 0.) -> Tensor:
        """
        Recommends embeddings based on the user profile, the number of recommendations and the trade-off between
        exploration and exploitation.
        :param user_profile: Low-dimensional user profile containing embeddings and preferences.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation. Higher beta means more exploitation.
            Must be in [0, 1]. 0 means full exploration, 1 means full exploitation.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        # Get embeddings and ratings from user profile
        embeddings, preferences = user_profile

        # Change Preference shape
        preferences = preferences.reshape(-1, 1)

        # Standardize embeddings
        mean, std = torch.mean(embeddings, dim=0), torch.std(embeddings, dim=0)
        embeddings_std = (embeddings - mean) / std

        # Build a search space for the BO to look in
        search_space = self.build_search_space()

        # Standardize search space
        search_space = (search_space - mean) / std

        # Get new acquisitions step by step
        for _ in range(n_recommendations):
            # Build a GP model
            mean_mod = ConstantMean()  # LinearMean(input_size=embeddings_std.shape[-1])
            model = SingleTaskGP(train_X=embeddings_std, train_Y=preferences, mean_module=mean_mod)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll = fit_gpytorch_mll(mll)

            # Initialize the acquisition function
            beta = 20 - get_unnormalized_value(beta, 0, 20)
            acqf = UpperConfidenceBound(model=model, beta=beta, maximize=True)

            # Get the highest scoring candidates out of meshgrid
            scores = acqf(search_space.reshape(search_space.shape[0], 1, search_space.shape[1]))
            candidate_idx = torch.argmax(scores)
            candidate = search_space[candidate_idx].reshape(1, -1)

            # Remove candidate from search space
            search_space = torch.cat((search_space[:candidate_idx], search_space[candidate_idx + 1:]))

            # Extend data with new candidate and predicted preference to include this information in the next iteration
            pseudo_preference = acqf._mean_and_sigma(X=candidate, compute_sigma=False)[0].detach()
            embeddings_std = torch.cat((embeddings_std, candidate))
            preferences = torch.cat((preferences, pseudo_preference.reshape(1, 1)))

        # Return most promising candidates
        candidates_std = embeddings_std[-n_recommendations:]

        # Unstandardize and return them
        candidates = candidates_std * std + mean
        return candidates

    def heat_map_values(self, user_profile: Tensor, user_space: Tensor, beta: float = 0.):
        """
        Get the values of the acquisition function for a given user profile and user space.
        These values can be used to create a heat map of the acquisition function, indicating which areas are likely to
        be chosen as future samples.
        :param user_profile: Low-dimensional user profile containing embeddings and preferences.
        :param user_space: The user space in which the user profile lies.
        :param beta: Trade-off between exploration and exploitation. Must be in [0, 1].
            0 means full exploration, 1 means full exploitation.
        :return: Tensor containing the values of the acquisition function.
        """
        # Get embeddings and ratings from user profile
        if user_profile is None:
            return None
        embeddings, preferences = user_profile

        # Change Preference shape
        preferences = preferences.reshape(-1, 1)

        # Standardize embeddings
        mean, std = torch.mean(embeddings, dim=0), torch.std(embeddings, dim=0)
        embeddings_std = (embeddings - mean) / std

        # Build a search space for the BO to look in
        search_space = user_space

        # Standardize search space
        search_space = (search_space - mean) / std

        # Build a GP model
        mean_mod = ConstantMean()  # LinearMean(input_size=embeddings_std.shape[-1])
        model = SingleTaskGP(train_X=embeddings_std, train_Y=preferences, mean_module=mean_mod)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = fit_gpytorch_mll(mll)

        # Initialize the acquisition function
        beta = 20 - get_unnormalized_value(beta, 0, 20)
        acqf = UpperConfidenceBound(model=model, beta=beta, maximize=True)

        # Get the highest scoring candidates out of meshgrid
        scores = acqf._mean_and_sigma(X=search_space.reshape(search_space.shape[0], 1, search_space.shape[1]),
                                      compute_sigma=False)[0].detach()

        return scores


class HypersphericalRandomRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis, seed: int = 42):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        # The coefficients of the embeddings and the latents must be normalized independently since they both rely on
        # the property of being unit length.

        # for embeddings
        embedding_coeffs = torch.randn(self.n_embedding_axis, n_recommendations, generator=self.generator)
        embedding_coeffs = embedding_coeffs / torch.linalg.norm(embedding_coeffs, dim=0, keepdim=True)

        # for latents
        latent_coeffs = torch.randn(self.n_latent_axis, n_recommendations, generator=self.generator)
        latent_coeffs = latent_coeffs / torch.linalg.norm(latent_coeffs, dim=0, keepdim=True)

        recommendation = torch.cat((embedding_coeffs, latent_coeffs), dim=0)
        return recommendation.T


class HypersphericalMovingCenterRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis, seed: int = 42):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        # for embeddings
        radius = (1 - beta) * 1.0  # todo make configurable
        center = user_profile[:self.n_embedding_axis, None]
        pos = center + torch.randn(self.n_embedding_axis, n_recommendations, generator=self.generator)
        moved_center = (1 - (radius ** 2) / 2) * center  # midpoint of the intersecting sphere
        # moved_center^T \cdot (moved_pos - moved_center) = 0 (orthogonality)
        moved_pos = pos * (moved_center.T @ moved_center) / (moved_center.T @ pos)
        dist = torch.linalg.norm(moved_pos - moved_center, dim=0, keepdim=True)
        embedding_recommendations = (moved_pos - moved_center) / dist * np.sqrt(
            radius ** 2 - radius ** 4 / 4) + moved_center

        # for latents
        radius = (1 - beta) * 1.0  # todo make configurable
        center = user_profile[-self.n_latent_axis:, None]
        pos = center + torch.randn(self.n_latent_axis, n_recommendations, generator=self.generator)
        moved_center = (1 - (radius ** 2) / 2) * center  # midpoint of the intersecting sphere
        # moved_center^T \cdot (moved_pos - moved_center) = 0 (orthogonality)
        moved_pos = pos * (moved_center.T @ moved_center) / (moved_center.T @ pos)
        dist = torch.linalg.norm(moved_pos - moved_center, dim=0, keepdim=True)
        latent_recommendations = (moved_pos - moved_center) / dist * np.sqrt(
            radius ** 2 - radius ** 4 / 4) + moved_center

        recommendations = torch.cat((embedding_recommendations, latent_recommendations), dim=0)
        return recommendations.T


class HypersphericalBayesianRecommender(Recommender):
    def __init__(self, n_embedding_axis, n_latent_axis, seed: int = 42):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis

    def sample_on_unit_sphere(self, n_samples: int, dim: int):
        x = torch.randn(n_samples, dim, generator=self.generator)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        train_X, train_Y = user_profile

        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y.reshape((-1, 1)),
            input_transform=None,
            # todo might be that Normalize(d=self.n_embedding_axis+self.n_latent_axis) is required here.
            outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        beta = 20 - get_unnormalized_value(beta, 0, 20)
        acqf = qUpperConfidenceBound(model=gp, beta=beta)

        # Sample many candidates
        candidates = torch.cat((self.sample_on_unit_sphere(10000, self.n_embedding_axis),
                                self.sample_on_unit_sphere(10000, self.n_latent_axis)), dim=-1)

        values = acqf(candidates.unsqueeze(1))  # shape [N, 1] for q=1
        topk = torch.topk(values.squeeze(), n_recommendations)

        recommendations = candidates[topk.indices]

        return recommendations
