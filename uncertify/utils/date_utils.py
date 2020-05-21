from datetime import datetime

DATE_FORMAT = '%Y-%m-%d'
DATE_TIME_FORMAT = '%Y-%m-%d_%H-%M-%S-%f'
TIME_FORMAT = '%H-%M-%S-%f'


def get_date_tag() -> str:
    """Returns a date string tag according to the DATE_FORMAT specifier,
    e.g. '2019-04-01'"""
    return datetime.now().strftime(DATE_FORMAT)


def get_date_time_tag() -> str:
    """Returns a datetime tag in the following example
    format '2019-04-01_20-24-13-535342'."""
    curr_time = datetime.now()
    return curr_time.strftime(DATE_TIME_FORMAT)


def get_time_tag() -> str:
    return datetime.now().strftime(TIME_FORMAT)


def load_datetime_from_string_tag(input_string) -> datetime:
    """Given a date time tag in string, parse the date time object."""
    return datetime.strptime(input_string, DATE_TIME_FORMAT)