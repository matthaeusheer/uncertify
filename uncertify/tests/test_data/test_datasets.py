from uncertify.data.datasets import train_val_split_list


def test_train_val_split_list():
    test_lists = list(range(10))
    train_list, val_list = train_val_split_list(test_lists, train_fraction=0.3)
    assert len(train_list) == 7
    assert len(val_list) == 3