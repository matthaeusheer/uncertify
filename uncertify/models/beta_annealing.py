import math
from dataclasses import dataclass

from uncertify.visualization.plotting import setup_plt_figure


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


def beta_config_factory(config_type: str,
                        beta_final: float = 1.0, beta_start: float = 0.0,
                        final_train_step: int = None, cycle_size: int = None,
                        cycle_size_const_fraction: float = 0.5) -> BetaConfig:
    assert config_type in ['monotonic', 'constant', 'cyclic'], f'Config type {config_type} not allowed.'
    if config_type == 'constant':
        return ConstantBetaConfig(beta_final)
    if config_type == 'monotonic':
        assert final_train_step is not None
        return MonotonicBetaConfig(final_train_step, beta_final, beta_start)
    if config_type == 'cyclic':
        assert cycle_size is not None
        assert cycle_size_const_fraction is not None
        return CyclicBetaConfig(cycle_size, cycle_size_const_fraction, beta_final, beta_start)


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


def cyclical_annealing(train_step: int, cycle_size: int, cycle_size_const_fraction: float = 0.5,
                       beta_final: float = 1.0, beta_start: float = 0.0) -> float:
    """Calculate beta (KL term weight) as a function of the train step with cyclic annealing.

    How to calculate the cycle size:

    The cyclic scheduling scheme looks like this. The "---" part is cycle_size_const_fraction from the
    whole cycle_size where beta = beta_final. Before a monotonic beta increase happens from beta_start to beta_final.
    beta_final           ---  ---  ---
                        /  | /  | /  |
    beta_start         /   |/   |/   | ...
                      |--------------|
                        cycle_size
    """
    first_const_step = math.ceil(cycle_size - cycle_size * cycle_size_const_fraction)  # within a cycle
    step_in_cycle = train_step % cycle_size
    if step_in_cycle < first_const_step:
        slope = (beta_final - beta_start) / first_const_step
        return slope * step_in_cycle + beta_start
    else:
        return beta_final


def plot_annealing_schedules(n_train_steps: int, cycle_size: int):
    # Monotonic annealing
    betas = [monotonic_annealing(train_step, n_train_steps // 2) for train_step in range(n_train_steps)]
    fig, ax = setup_plt_figure(figsize=(12, 6), title='Monotonic annealing schedule',
                               xlabel='Training step', ylabel=r'$\beta$')
    ax.plot(range(n_train_steps), betas, '-', linewidth=2)

    # Cyclical annealing
    betas = [cyclical_annealing(train_step, cycle_size) for train_step in range(n_train_steps)]
    fig, ax = setup_plt_figure(figsize=(12, 6), title='Cyclical annealing schedule',
                               xlabel='Training step', ylabel=r'$\beta$')
    ax.plot(range(n_train_steps), betas, '-', linewidth=2)
