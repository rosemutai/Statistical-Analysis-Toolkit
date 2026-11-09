import csv
import math
from collections import Counter
import numpy as np

def mean(num_list):
    if not num_list:
        return None
    return sum(num_list) / len(num_list)

def median(num_list):
    if not num_list:
        raise ValueError('Enter a non empty list')
    sorted_list = sorted(num_list)
    list_length = len(sorted_list)

    middle = list_length // 2

    if list_length % 2 != 0:
        return sorted_list[middle]
    return (sorted_list[middle - 1] + sorted_list[middle]) / 2

def mode(num_list):
    if not num_list:
        raise ValueError('Enter a non empty list')

    counts = {}
    for item in num_list:
        counts[item] = counts.get(item, 0) + 1

    highest_frequency = max(counts.values())

    if highest_frequency == 1:
        return []
    return [item for item, frequency in counts.items() if frequency == highest_frequency]

def variance(values, sample):
    if not values:
            raise ValueError('Enter a non empty list')

    values_size = len(values)

    mean = sum(values) / values_size
    x_bar = [(value - mean)**2 for value in values]

    denominator = values_size - 1 if sample else values_size

    return sum(x_bar) / denominator


def std_dev(values, sample):
    return variance(values, sample) ** 0.5


def load_column(path, column):
    valid_floats = []
    rejected_list = []

    missing_values = ['', 'na', 'n/a', 'null', 'none', 'nan', None]

    with open(path, 'r') as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f'Column {column} not found in the file')

        for row_num, row in enumerate(reader, start=2):
            row_value = row.get(column)

            if row_value.strip().lower() in missing_values:
                reason = 'Missing Value'

            else:
                try:
                    value = float(row_value.strip())
                except (ValueError):
                    reason = 'Invalid value'
                else:
                    valid_floats.append(value)
                    continue
            rejected_list.append((row_num, row_value))
        return valid_floats, rejected_list


def quartiles(values):
    sorted_values = sorted(values)

    values_size = len(sorted_values)

    mid = values_size // 2
    first_half = sorted_values[:mid]
    last_half = sorted_values[(mid+1 if values_size % 2 else mid):]

    q1 = median(first_half)
    q2 = median(sorted_values)
    q3 = median(last_half)

    return q1, q2, q3


def iqr_outliers(values, k):
    q1, q2, q3 = quartiles(values)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outliers = [
        value for value in values if value < lower_bound or value > upper_bound
    ]

    return {
        'IQR': iqr,
        'minimum_value': lower_bound,
        'maximum_value': upper_bound,
        'outliers': outliers
    }

class StatisticsAnalyzer:
    def __init__(self, values, rejected=None):
        if not values:
            raise ValueError('Cannot analyze an empty dataset')
        
        self.values = list(values)
        self.rejected = rejected
        self.sorted_values = sorted(self.values)
        self._summary = None

    @classmethod
    def from_csv(cls, path, col):
        values, rejected = load_column(path, col)

        return cls(values, rejected)

    def summary (self):
        if self._summary is None:
            q1, q2, q3 = quartiles(self.sorted_values)
            iqr = q3 - q1
            outliers = iqr_outliers(self.sorted_values, 1.5)

            self._summary = {
                'count': len(self.values),
                'mean': mean(self.values),
                'median': median(self.sorted_values),
                'mode': mode(self.values),
                'variance': variance(self.values, False),
                'std_dev': std_dev(self.values, False),
                'minimum': self.sorted_values[0],
                'maximum': self.sorted_values[-1],
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'outliers': outliers               
            }
        return self._summary
    
    def without_outliers (self, k=1.5):
        outliers = iqr_outliers(self.sorted_values, k)

        cleaned_values = [
            value for value in self.values if value not in outliers
        ]
        return StatisticsAnalyzer(cleaned_values, self.rejected)

    def report (self):
        stats = self.summary()
        print('\nStatistics Report')
        print('-'*20)

        for name, value in stats.items():
            print(f'{name}: {value}')
        print(f'Rejected rows: {len(self.rejected)}')


def compare_with_numpy(analyzer):
    import numpy as np

    stats = analyzer.summary()
    values = np.array(analyzer.values, dtype=float)
    sorted_values = np.sort(values)

    numpy_results = {
        'count': len(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'variance': np.var(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'q1': np.percentile(values, 25),
        'q3': np.percentile(values, 75)
    }

    numpy_results['iqr'] = numpy_results['q3'] - numpy_results['q1']
    unique_values, counts = np.unique(values, return_counts=True)

    highest_frequency = np.max(counts)
    numpy_results['mode'] = unique_values[counts == highest_frequency].tolist()

    print('\nNumpy Verification Report')
    print('-' * 20)

    for name, numpy_value in numpy_results.items():
        custom_value = stats[name]

        if custom_value is None or numpy_value is None:
            match = custom_value is None and numpy_value is None
        elif name == 'mode':
            custom_modes = np.asarray(custom_value, dtype=float)
            numpy_modes = np.asarray(custom_value, dtype=float)

            match = (
                len(custom_modes) == len(numpy_modes) and
                np.allclose(
                    np.sort(custom_modes),
                    np.sort(numpy_modes),
                    rtol=1e-5,
                    atol=1e-8
                )
            )
        else:
            np.isclose(
                custom_value,
                numpy_value,
                rtol=1e-5,
                atol=1e-8
            )
        status = 'MATCH' if match else 'MISMATCH'
        print(
            f'{name:12}: {status} | '
            f'Custom = {custom_value} | '
            f'NumPy = {numpy_value}'
        )