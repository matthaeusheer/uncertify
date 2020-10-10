from torch.utils.data import DataLoader

from uncertify.models.beta_annealing import monotonic_annealing
from uncertify.tests.test_data.dummy_dataset import DummyDataSet


def test_monotonic_annealing():
    """Note: See the plot_annealing_schedules function to check whether your annealing schedule works."""
    data_loader = DataLoader(dataset=DummyDataSet(n_samples=100), batch_size=10)
    n_batches_per_epoch = len(data_loader.dataset) // data_loader.batch_size
    n_epochs = 5
    n_epochs_to_final = 3
    beta_start = 0.0
    beta_final = 1.0

    final_train_step = n_epochs_to_final * n_batches_per_epoch

    for train_step in range(n_batches_per_epoch * n_epochs):
        beta = monotonic_annealing(train_step, final_train_step, beta_final, beta_start)
        if train_step == 0:
            assert beta == beta_start
        if train_step >= final_train_step:
            assert beta == beta_final
        else:
            assert beta_start <= beta < beta_final
        # Uncomment if you want to see the beta values for the settings above. Uncomment to not pollute test-suite.
        # print(f'{train_step} (epoch {(train_step) // n_batches_per_epoch}): {round(beta, 2)}')

