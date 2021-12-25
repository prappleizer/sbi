# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.distributions import Bernoulli
from torch.nn.utils import clip_grad_norm_
from torch.utils import data

from sbi import utils as utils
from sbi.inference.snle.snle_base import LikelihoodEstimator
from sbi.types import TorchModule
from sbi.utils import validate_theta_and_x, x_shape_from_simulation
from sbi.utils.sbiutils import mask_sims_from_prior
from sbi.types import TensorboardSummaryWriter
from sbi.utils import del_entries


class MNLE(LikelihoodEstimator):
    def __init__(
        self,
        prior,
        density_estimator: Union[str, Callable] = "nsf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
        **unused_args,
    ):
        r"""Mixed Neural Likelihood Estimation [MNLE, 1]

        MNLE extende NLE to simulator with mixed data output, e.g., discrete choices
        and continuous reaction times as often used in models of decision-making.
        It trains two separate density estimator, a Bernoulli net for the discrete
        data, and a neural spline flow for the continuous data.

        [1] https://www.biorxiv.org/content/10.1101/2021.12.22.473472v1

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
            unused_args: Absorbs additional arguments. No entries will be used. If it
                is not empty, we warn. In future versions, when the new interface of
                0.14.0 is more mature, we will remove this argument.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__", "unused_args"))
        super().__init__(**kwargs, **unused_args)

        # MNLE-specific summary_writer fields.
        self._summary.update({"rt_net_epochs": []})  # type: ignore
        self._summary.update({"rt_net_best_validation_log_probs": []})  # type: ignore
        self._summary.update({"rt_net_validation_log_probs": []})  # type: ignore
        self._summary.update({"choice_net_epochs": []})  # type: ignore
        self._summary.update({"choice_net_best_validation_log_probs": []})  # type: ignore
        self._summary.update({"choice_net_validation_log_probs": []})  # type: ignore

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        from_round: int = 0,
    ) -> "LikelihoodEstimator":
        r"""
        Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `SNLE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.

        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        theta, x = validate_theta_and_x(theta, x, training_device=self._device)
        assert x.shape[1] == 2, "MNLE assumes x to have two columns: [rts, choices]."
        assert from_round == 0, "MNLE is not built for inference in multiple rounds."

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(mask_sims_from_prior(int(from_round), theta.size(0)))
        self._data_round_index.append(int(from_round))

        return self

    def train(
        self,
        architecture_hyperparameters: Dict,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        exclude_invalid_x: bool = True,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> nn.Module:
        r"""
        Train the density estimator to learn the distribution $p(x|\theta)$.

        Args:
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that has learned the distribution $p(x|\theta)$.
        """

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)
        # Load data from most recent round.
        self._round = max(self._data_round_index)
        theta, x, _ = self.get_simulations(
            start_idx, exclude_invalid_x, warn_on_invalid=True
        )

        self._x_shape = x_shape_from_simulation(x)
        rts = abs(x[:, :1])
        choices = torch.ones_like(rts)
        choices[x[:, 0] < 0, :] = 0
        # Concatenate theta and choices for conditional flow training below.
        theta_and_choices = torch.cat((theta, choices), dim=1)

        # Dataset is shared for training and validation loaders.
        # Separate datasets for choice net training.
        choice_dataset = data.TensorDataset(theta, choices)
        # And rts-flow training, conditioning variable contains parameters and choices.
        rt_dataset = data.TensorDataset(theta_and_choices, rts)

        # Train rt-flow first.
        rt_train_loader, rt_val_loader = self.get_dataloaders(
            rt_dataset,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )
        rt_net = self._build_neural_net(
            theta_and_choices[self.train_indices], rts[self.train_indices]
        )
        assert (
            len(self._x_shape) < 3
        ), "SNLE cannot handle multi-dimensional simulator output."

        rt_net = self.train_net(
            rt_net,
            rt_train_loader,
            rt_val_loader,
            logging_net_string="rt_net",
            learning_rate=learning_rate,
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            resume_training=resume_training,
            clip_max_norm=clip_max_norm,
        )

        # Train choice net independently.
        choice_train_loader, choice_val_loader = self.get_dataloaders(
            choice_dataset,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )
        choice_net = BernoulliMN(
            n_input=theta.shape[1],
            n_hidden_units=architecture_hyperparameters["num_hidden_units"],
            n_hidden_layers=architecture_hyperparameters["num_hidden_layers"],
        )

        choice_net = self.train_net(
            choice_net,
            choice_train_loader,
            choice_val_loader,
            logging_net_string="choice_net",
            learning_rate=learning_rate,
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            resume_training=resume_training,
            clip_max_norm=clip_max_norm,
        )

        # Update TensorBoard and summary dict.
        self._summarize(
            round_=self._round,
            x_o=None,
            theta_bank=theta,
            x_bank=x,
        )

        # Build mixed synthetic likelihood.
        self._neural_net = MixedSyntheticLikelihood(
            choice_net,
            rt_net,
        )

        return deepcopy(self._neural_net)

    def train_net(
        self,
        net,
        train_loader,
        val_loader,
        learning_rate,
        max_num_epochs,
        stop_after_epochs,
        logging_net_string,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        show_train_summary: bool = False,
    ):
        """Train a neural net by maximizing its log-prob of the data."""

        net.to(self._device)
        self._neural_net = net
        if not resume_training:
            self.optimizer = optim.Adam(
                list(net.parameters()),
                lr=learning_rate,
            )
        self.epoch, self._val_log_prob = 0, float("-Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):

            # Train for a single epoch.
            net.train()
            for batch in train_loader:
                self.optimizer.zero_grad()
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                # Evaluate on x with theta as context.
                log_prob = net.log_prob(x_batch, theta_batch)
                loss = -torch.mean(log_prob)
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        net.parameters(),
                        max_norm=clip_max_norm,
                    )
                self.optimizer.step()

            self.epoch += 1

            # Calculate validation performance.
            net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    # Evaluate on x with theta as context.
                    log_prob = net.log_prob(x_batch, theta_batch)
                    log_prob_sum += log_prob.sum().item()
            # Take mean over all validation samples.
            self._val_log_prob = log_prob_sum / (
                len(val_loader) * val_loader.batch_size
            )
            # Log validation log prob for every epoch.
            self._summary[f"{logging_net_string}_validation_log_probs"].append(
                self._val_log_prob
            )

            self._maybe_show_progress(
                logging_net_string, self._show_progress_bars, self.epoch
            )

        # Update summary.
        self._summary[f"{logging_net_string}_epochs"].append(self.epoch)
        self._summary[f"{logging_net_string}_best_validation_log_probs"].append(
            self._best_val_log_prob
        )
        self._summary["epochs"].append(self.epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        return deepcopy(net)

    @staticmethod
    def _maybe_show_progress(net_str=str, show=bool, epoch=int) -> None:
        if show:
            # end="\r" deletes the print statement when a new one appears.
            # https://stackoverflow.com/questions/3419984/
            print(f"Training {net_str} network. Epochs trained: ", epoch, end="\r")


class BernoulliMN(nn.Module):
    """Net for learning a conditional Bernoulli mass function over choices given parameters.

    Takes as input parameters theta and learns the parameter p of a Bernoulli.

    Defines log prob and sample functions.
    """

    def __init__(self, n_input=4, n_output=1, n_hidden_units=20, n_hidden_layers=2):
        """Initialize Bernoulli mass network.

        Args:
            n_input: number of input features
            n_output: number of output features, default 1 for a single Bernoulli variable.
            n_hidden_units: number of hidden units per hidden layer.
            n_hidden_layers: number of hidden layers.
        """
        super(BernoulliMN, self).__init__()

        self.n_hidden_layers = n_hidden_layers

        self.activation_fun = nn.Sigmoid()

        self.input_layer = nn.Linear(n_input, n_hidden_units)

        # Repeat hidden units hidden layers times.
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))

        self.output_layer = nn.Linear(n_hidden_units, n_output)

    def forward(self, theta):
        """Return Bernoulli probability predicted from a batch of parameters.

        Args:
            theta: batch of input parameters for the net.

        Returns:
            Tensor: batch of predicted Bernoulli probabilities.
        """
        assert theta.dim() == 2, "theta needs to have a batch dimension."

        # forward path
        theta = self.activation_fun(self.input_layer(theta))

        # iterate n hidden layers, input x and calculate tanh activation
        for layer in self.hidden_layers:
            theta = self.activation_fun(layer(theta))

        p_hat = self.activation_fun(self.output_layer(theta))

        return p_hat

    def log_prob(self, x, theta):
        """Return Bernoulli log probability of choices x, given parameters theta.

        Args:
            x: choices to evaluate.
            theta: parameters for input to the BernoulliMN.

        Returns:
            Tensor: log probs with shape (x.shape[0],)
        """
        # Predict Bernoulli p and evaluate.
        p = self.forward(theta=theta)
        return Bernoulli(probs=p).log_prob(x)

    def sample(self, theta, num_samples):
        """Returns samples from Bernoulli RV with p predicted via net.

        Args:
            theta: batch of parameters for prediction.
            num_samples: number of samples to obtain.

        Returns:
            Tensor: Bernoulli samples with shape (batch, num_samples, 1)
        """

        # Predict Bernoulli p and sample.
        p = self.forward(theta)
        return Bernoulli(probs=p).sample((num_samples,))


class MixedSyntheticLikelihood(nn.Module):
    """Class for Mixed Neural Likelihood Estimation. It combines a Bernoulli choice
    net and a flow over reaction times to model decision-making data."""

    def __init__(
        self, choice_net: nn.Module, rt_net: nn.Module, use_log_rts: bool = False
    ):
        """Initializa synthetic likelihood class from a choice net and reaction time
        flow.

        Args:
            choice_net: BernoulliMN net trained to predict choices from DDM parameters.
            rt_net: generative model of reaction time given DDM parameters and choices.
            use_log_rts: whether the rt_net was trained with reaction times transformed
                to log space.
        """
        super(MixedSyntheticLikelihood, self).__init__()

        self.choice_net = choice_net
        self.rt_net = rt_net
        self.use_log_rts = use_log_rts

    def sample(self, theta, num_samples: int = 1, track_gradients=False):
        """Return choices and reaction times given DDM parameters.

        Args:
            theta: DDM parameters, shape (batch, 4)
            num_samples: number of samples to generate.

        Returns:
            Tensor: samples data (rt, choice) with shape (num_samples, 2)
        """
        assert theta.shape[0] == 1, "for samples, no batching in theta is possible yet."

        with torch.set_grad_enabled(track_gradients):

            # Sample choices given parameters, from BernoulliMN.
            choices = self.choice_net.sample(theta, num_samples).reshape(num_samples, 1)
            # Pass num_samples=1 because the choices in the context contains num_samples elements already.
            samples = self.rt_net.sample(
                num_samples=1,
                # repeat the single theta to match number of sampled choices.
                context=torch.cat((theta.repeat(num_samples, 1), choices), dim=1),
            ).reshape(num_samples, 1)
        return samples.exp() if self.use_log_rts else samples, choices

    def log_prob(
        self,
        x,
        theta,
        track_gradients=False,
        ll_lower_bound=np.log(1e-7),
    ):
        """Return log likelihood for each entry in a batch of data (rts and choices),
        and parameters theta.

        The batch size of x and theta must match, x is not assumend to contain iid
        trials.

        Args:
            x: data tensor containing two columns, one with reaction times and one
                with choices.
            theta: parameters
            track_gradients: Whether to track the gradients when evaluating the nets.
            ll_lower_bound: Log-likelihood value to use as lower bound.

        Returns:
            Log-likelihoods for each entry in the batch of x and theta.
        """
        num_parameters = theta.shape[0]
        # Parameters can be repeated
        theta_unique, theta_unique_inverse = theta.unique(
            return_inverse=True,
            sorted=False,
            dim=0,
        )
        num_unique_parameters = theta_unique.shape[0]
        assert x.ndim > 1
        assert x.shape[1] == 2, "MNLE assumes x to have two columns: [rts; choices]"
        rts = x[:, 0:1]
        choices = x[:, 1:2]
        assert (
            x.shape[0] == theta.shape[0]
        ), "Input and context must have same batch shape."

        with torch.set_grad_enabled(track_gradients):
            # Get choice log probs from choice net.
            # There are only two choices, thus we only have to get the log probs of those.
            # (We could even just calculate one and then use the complement.)
            zero_choice = torch.zeros(1, 1)
            # Calculate zero choice lps for unique parameters.
            zero_choice_lp_unique = self.choice_net.log_prob(
                torch.repeat_interleave(zero_choice, num_unique_parameters, dim=0),
                theta_unique,
            ).reshape(1, -1)
            # Expand to parameters.
            zero_choice_lp = zero_choice_lp_unique[theta_unique_inverse]

            # Calculate complement one-choice log prob.
            one_choice_lp = torch.log(1 - zero_choice_lp.exp())
            zero_one_lps = torch.cat((zero_choice_lp, one_choice_lp), dim=1)

            # For each choice choose the corresponding lp.
            lp_choices = zero_one_lps[
                torch.arange(num_parameters),
                torch.as_tensor(choices, dtype=int).squeeze(),
            ]

            # Get rt log probs from rt net.
            lp_rts = self.rt_net.log_prob(
                rts,
                context=torch.cat((theta, choices), dim=1),
            )

        # Combine into joint lp with first dim over trials.
        lp_combined = lp_choices + lp_rts

        # Maybe add log abs det jacobian of RTs: log(1/rt) = - log(rt)
        if self.use_log_rts:
            lp_combined -= torch.log(rts)

        # Set to lower bound where reaction happend before non-decision time tau.
        lp = torch.where(
            torch.logical_and(
                # If rt < tau the likelihood should be zero (i.e., set to lower bound).
                (rts > theta[:, -1:]).squeeze(),
                # Apply lower bound.
                lp_combined > ll_lower_bound,
            ),
            lp_combined,
            ll_lower_bound * torch.ones_like(lp_combined),
        )

        # Return sum over iid trial log likelihoods.
        return lp
