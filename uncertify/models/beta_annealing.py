import math
import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt

from uncertify.models.utils import generalized_logistic_curve
from uncertify.visualization.plotting import setup_plt_figure

LOG = logging.getLogger(__name__)


@dataclass
class BetaConfig:
    pass


@dataclass
class ConstantBetaConfig(BetaConfig):
    beta: float


@dataclass
class MonotonicBetaConfig(BetaConfig):
    final_train_step: int
    beta_final: float
    beta_start: float


@dataclass
class CyclicBetaConfig(BetaConfig):
    cycle_size: int
    cycle_size_const_fraction: float
    beta_final: float
    beta_start: float


@dataclass
class SigmoidBetaConfig(BetaConfig):
    beta_final: float
    beta_start: float


@dataclass
class DecayBetaConfig(BetaConfig):
    final_train_step: int
    beta_final: float
    beta_start: float


def beta_config_factory(config_type: str,
                        beta_final: float = 1.0, beta_start: float = 0.0,
                        final_train_step: int = None, cycle_size: int = None,
                        cycle_size_const_fraction: float = 0.5) -> BetaConfig:
    assert config_type in ['monotonic', 'constant', 'cyclic',
                           'sigmoid', 'decay'], f'Config type {config_type} not allowed.'
    if config_type == 'constant':
        return ConstantBetaConfig(beta_final)
    if config_type == 'monotonic':
        assert final_train_step is not None
        return MonotonicBetaConfig(final_train_step, beta_final, beta_start)
    if config_type == 'cyclic':
        assert cycle_size is not None
        assert cycle_size_const_fraction is not None
        return CyclicBetaConfig(cycle_size, cycle_size_const_fraction, beta_final, beta_start)
    if config_type == 'sigmoid':
        LOG.warning(f'Created sigmoid annealing config with hardcoded parameters.')
        return SigmoidBetaConfig(beta_final, beta_start)
    if config_type == 'decay':
        return DecayBetaConfig(final_train_step, beta_final, beta_start)


def monotonic_annealing(train_step: int, final_train_step: int,
                        beta_final: float = 1.0, beta_start: float = 0.0) -> float:
    """Calculate beta (KL term weight) as a function of training step for monotonic, linear annealing.

    How to calculate the train step?
        train_steps_per_epoch = n_batches_per_epoch
        train_step_global = n_batches_per_epoch * current_epoch
    Example:
        You have 10'000 train images, a batch size of 64 and want to have beta go from 0 to 1 in 10 epochs.
        n_batches_per_epoch = ceil(10'000 / 64) = 157
        final_train_step = n_batches_per_epoch (157) * n_epochs_to_final (10) = 1570

    Arguments:
        train_step: the global training step
        final_train_step: the training step when beta_final will be reached
        beta_final: the final value of beta, when this value is reached it will stay at this value for ever on forward
        beta_start: starting value of beta
    """
    if train_step > final_train_step:
        return beta_final
    else:
        slope = (beta_final - beta_start) / final_train_step
        return slope * train_step + beta_start


def sigmoid_annealing(train_step: int, beta_final: float = 1.0, beta_start: float = 0.0) -> float:
    """Similar to monotonic annealing (check for Arguments) but instead of a monotonic increase us a sigmoid shape."""
    # CAUTION: The parameters here are HARDCODED, if you need another behaviour, change it in code.
    return generalized_logistic_curve(train_step, a=beta_start, k=beta_final, b=0.01, q=0.5, eta=0.05, c=1)


def cyclical_annealing(train_step: int, cycle_size: int, cycle_size_const_fraction: float = 0.5,
                       beta_final: float = 1.0, beta_start: float = 0.0) -> float:
    """Calculate beta (KL term weight) as a function of the train step with cyclic annealing.

    How to calculate the cycle size:

    The cyclic scheduling scheme looks like this. The "---" part is cycle_size_const_fraction from the
    whole cycle_size where beta = beta_final. Before a monotonic beta increase happens from beta_start to beta_final.
    beta_final           ---  ---  ---
                        /  | /  | /  |
    beta_start         /   |/   |/   | ...
                      |----|
                    cycle_size
    """
    first_const_step = math.ceil(cycle_size - cycle_size * cycle_size_const_fraction)  # within a cycle
    step_in_cycle = train_step % cycle_size
    if step_in_cycle < first_const_step:
        slope = (beta_final - beta_start) / first_const_step
        return slope * step_in_cycle + beta_start
    else:
        return beta_final


def decay_annealing(train_step: int, final_train_step: int,
                    beta_final: float = 1.0, beta_start: float = 0.0) -> float:
    """Exponential decay annealing such that at final_train_step and from there on beta is beta_final."""
    if train_step < final_train_step:
        decay_rate = math.log(beta_final / beta_start) / final_train_step
        return beta_start * math.exp(decay_rate * train_step)
    else:
        return beta_final


def plot_annealing_schedules(n_train_steps: int, cycle_size: int) -> plt.Figure:
    fig, ax = setup_plt_figure(figsize=(11, 5), title='Annealing schedules compared',
                               xlabel='Training step', ylabel=r'$\beta$')
    monotonic_betas = [monotonic_annealing(train_step, n_train_steps // 2) for train_step in range(n_train_steps)]
    ax.plot(range(n_train_steps), monotonic_betas, '-', linewidth=2, label='monotonic')

    cyclic_betas = [cyclical_annealing(train_step, cycle_size, beta_final=1.025) for train_step in range(n_train_steps)]
    ax.plot(range(n_train_steps), cyclic_betas, '-', linewidth=2, label='cyclic')

    sigmoid_betas = [sigmoid_annealing(train_step, beta_final=1.05) for train_step in range(n_train_steps)]
    ax.plot(range(n_train_steps), sigmoid_betas, '-', linewidth=2, label='sigmoid')

    decay_betas = [decay_annealing(train_step, n_train_steps // 5, beta_final=1.1, beta_start=2)
                   for train_step in range(n_train_steps)]
    ax.plot(range(n_train_steps), decay_betas, '-', linewidth=2, label='exponential decay')

    ax.legend(frameon=False)
    return fig
