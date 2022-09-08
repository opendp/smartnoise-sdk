import numpy as np
import pandas as pd
from collections import defaultdict


def gen_data_frame_with_schema(schema, n_records):
    return pd.DataFrame({
        k: list(np.random.choice(schema[k], size=n_records)) for k in schema
    }, columns=schema.keys())


def gen_data_frame(number_of_records_to_generate):
    return pd.concat([
        gen_data_frame_with_schema(
            {
                'H1': ['1', '2', ''],
                'H2': ['1', '2', '3', ''],
                'H3': ['1', '2', '3', '4', '5', ''],
                'H4': ['0', '1'],
                'H5': ['0', '1'],
                'H6': ['0', '1'],
                'H7': ['0', '1'],
                'H8': ['0', '1'],
                'H9': ['0', '1'],
                'H10': ['0', '1'],
            },
            number_of_records_to_generate // 2
        ),
        gen_data_frame_with_schema(
            {
                'H1': ['1', '2', ''],
                'H2': ['4', '5', '6', ''],
                'H3': ['6', '7', '8', '9', '10', ''],
                'H4': ['0', '1'],
                'H5': ['0', '1'],
                'H6': ['0', '1'],
                'H7': ['0', '1'],
                'H8': ['0', '1'],
                'H9': ['0', '1'],
                'H10': ['0', '1'],
            },
            number_of_records_to_generate // 2
        ),
    ], ignore_index=True)


class ErrorReport:
    def __init__(self, src_aggregates, target_aggregates):
        self.src_aggregates = src_aggregates
        self.target_aggregates = target_aggregates

    def calc_fabricated(self):
        self.fabricated_count = 0
        self.fabricated_count_by_len = defaultdict(int)

        for u in self.target_aggregates.keys():
            if u not in self.src_aggregates:
                self.fabricated_count += 1
                self.fabricated_count_by_len[len(u)] += 1

    def calc_suppressed(self):
        self.suppressed_count = 0
        self.suppressed_count_by_len = defaultdict(int)

        for o in self.src_aggregates.keys():
            if o not in self.target_aggregates:
                self.suppressed_count += 1
                self.suppressed_count_by_len[len(o)] += 1

    def calc_mean(self):
        mean = []
        mean_by_len = defaultdict(list)

        for o in self.src_aggregates.keys():
            mean.append(self.src_aggregates[o])
            mean_by_len[len(o)].append(self.src_aggregates[o])

        self.mean_count = np.mean(mean)
        self.mean_count_by_len = {
            l: np.mean(mean_by_len[l]) for l in mean_by_len.keys()
        }

    def calc_errors(self):
        errors = []
        errors_by_len = defaultdict(list)

        for o in self.src_aggregates.keys():
            if o in self.target_aggregates:
                err = abs(self.target_aggregates[o] - self.src_aggregates[o])
                errors.append(err)
                errors_by_len[len(o)].append(err)

        self.mean_error = np.mean(errors)
        self.mean_error_by_len = {
            l: np.mean(errors_by_len[l])
            for l in errors_by_len.keys()
        }

    def calc_total(aggregates):
        total = 0
        total_by_len = defaultdict(int)

        for o in aggregates.keys():
            total += 1
            total_by_len[len(o)] += 1

        return (total, total_by_len)

    def gen(self):
        self.calc_fabricated()
        self.calc_suppressed()
        self.calc_mean()
        self.calc_errors()
        self.src_total, self.src_total_by_len = ErrorReport.calc_total(
            self.src_aggregates)
        self.target_total, self.target_total_by_len = ErrorReport.calc_total(
            self.target_aggregates)

        rows = [
            [
                l,
                f'{self.mean_count_by_len[l]:.2f} +/- {self.mean_error_by_len[l]:.2f}',
                f'{self.suppressed_count_by_len[l] * 100.0 / self.src_total_by_len[l]:.2f} %',
                f'{self.fabricated_count_by_len[l] * 100.0 / self.target_total_by_len[l]:.2f} %',
            ] for l in sorted(self.mean_error_by_len.keys())
        ]
        rows.append([
            'All',
            f'{self.mean_count:.2f} +/- {self.mean_error:.2f}',
            f'{self.suppressed_count * 100.0 / self.src_total:.2f} %',
            f'{self.fabricated_count * 100.0 / self.target_total:.2f} %',
        ])

        return pd.DataFrame(rows, columns=[
            'Length',
            'Count +/- Error',
            'Suppressed %',
            'Fabricated %',
        ])
