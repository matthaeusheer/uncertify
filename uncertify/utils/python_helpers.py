def bool_to_str(boolean: bool) -> str:
    """Converts a bool such as True to 'true'."""
    return 'true' if boolean else 'false'


def get_indices_of_n_largest_items(input_list: list, n: int) -> list:
    return sorted(range(len(input_list)), key=lambda sub: input_list[sub])[-n:]


def get_indices_of_n_smallest_items(input_list: list, n: int) -> list:
    return sorted(range(len(input_list)), key=lambda sub: input_list[sub])[:n]


def get_idx_of_closest_value(some_list: list, value: float) -> int:
    return min(range(len(some_list)), key=lambda i: abs(some_list[i] - value))
