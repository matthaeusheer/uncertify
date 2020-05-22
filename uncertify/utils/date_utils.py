from datetime import datetime

DATE_FORMAT = '%Y-%m-%d'
DATE_TIME_FORMAT = '%Y-%m-%d_%H-%M-%S-%f'
TIME_FORMAT = '%H-%M-%S-%f'


def get_date_tag(date_format: str = DATE_FORMAT) -> str:
    """Returns a date string tag according to the DATE_FORMAT specifier,
    e.g. '2019-04-01'"""
    return datetime.now().strftime(date_format)


def get_date_time_tag(date_time_format: str = DATE_TIME_FORMAT) -> str:
    """Returns a datetime tag in the following example
    format '2019-04-01_20-24-13-535342'."""
    curr_time = datetime.now()
    return curr_time.strftime(date_time_format)


def get_time_tag(time_format: str = TIME_FORMAT) -> str:
    """Get current time in default format, e.g. 20-24-13-535342."""
    return datetime.now().strftime(time_format)


def load_datetime_from_string_tag(input_string, date_time_format) -> datetime:
    """Given a date time tag in string, parse the date time object."""
    return datetime.strptime(input_string, date_time_format)
