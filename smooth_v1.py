import pandas as pd
import numpy as np
import warnings
import argparse

class Smoothing:
    def __init__(self, csv, prediction_idx=0, group_idx=1, threshold=0.75, window_size=None, smoothing_type = 'mode'):
        self.csv = csv
        self.prediction_idx = prediction_idx
        self.group_idx = group_idx
        self.threshold = threshold
        self.window_size = window_size
        self.smoothing_type = smoothing_type
        self.smoothing_funcs = SmoothingFunctions(self.window_size, self.threshold)
    
    def smooth_series(self, values):
        if self.smoothing_type == 'mode':
            values = self.smoothing_funcs.mode(values)
        
        elif self.smoothing_type == 'mean':
            values = self.smoothing_funcs.mean(values)

        elif self.smoothing_type == 'median':
            values = self.smoothing_funcs.median(values)

        return values
    
    def smooth_csv_sample_level(self, csv, window_size=None, threshold=0.75, prediction_idx=0):
        raw_values = pd.read_csv(csv, header=0)
        # If only a single column is passed, make it iterable
        if isinstance(prediction_idx, int):
            prediction_idx = [prediction_idx]
        # Smooth columns
        for idx in prediction_idx:
            assert idx < len(raw_values.columns), "Column index provided is out of range!"
            raw_values[f"{raw_values.columns[idx]} (sample-smoothed)"] = self.smooth_series(raw_values[raw_values.columns[idx]], window_size=window_size, threshold=threshold)
        return raw_values
    
    def smooth_csv_group_level(self, csv, threshold=0.75, prediction_idx=0, group_idx=1):
        raw_values = pd.read_csv(csv, header=0)
        # If only a single column is passed, make it iterable
        if isinstance(prediction_idx, int):
            prediction_idx = [prediction_idx]
        
        # Smooth columns
        for idx in prediction_idx:
            assert idx < len(raw_values.columns), "Column index provided is out of range!"
            groups = raw_values.groupby([raw_values.columns[group_idx]])
            smooth_group = pd.Series([])
            for group in groups:
                smooth_group = pd.concat([smooth_group, self.smooth_series(group[1][idx], window_size=None, threshold=threshold)])
            raw_values[f"{raw_values.columns[idx]} (group-smoothed)"] = smooth_group
        return raw_values
    
    def save_sample_level_csv(self):
        smooth_df = self.smooth_csv_sample_level(self.csv, window_size=self.window_size, threshold=self.threshold, prediction_idx=self.prediction_idx)
        smooth_df.to_csv(f"smooth_{self.csv}")

    def save_group_csv(self):
        smooth_df = self.smooth_csv_group_level(self.csv, threshold=self.threshold, prediction_idx=self.prediction_idx, group_idx=self.group_idx)
        smooth_df.to_csv(f"smooth_{self.csv}")

class SmoothingFunctions:

    def __init__(self, window_size:int, threshold=0.75):
        self.mode_threshold = threshold
        self.window_size = window_size
    
    def mode(self, values:pd.Series):
        if self.window_size is None or self.window_size == 0 or self.window_size >= len(values):
            if self.window_size is not None and self.window_size > len(values):
                warnings.warn('Specified window size is bigger than series length... defaulting to file level smoothing.')
            modes = values.mode()[0] #sp.mode(values)[0][0]
            mode_percent = values.value_counts(1)[modes]
            if mode_percent >= self.mode_threshold:
                values.values[:] = modes

        else:
            # Create windows
            windows = values.rolling(window=self.window_size)
            modes = windows.apply(lambda x: x.mode()[0])
            mode_percent = windows.apply(lambda x: x.value_counts(1)[x.mode()[0]])
            # For each mode change the original value if threshold for percentage is met
            for i, window_mode in enumerate(modes):
                if not np.isnan(mode_percent[i]) and mode_percent[i] >= self.mode_threshold:
                    values[i] = window_mode
                else:
                    values[i] = values[i]
        return values
    
    def median(self, values:pd.Series):
        if self.window_size is None or self.window_size == 0 or self.window_size >= len(values):
            if self.window_size is not None and self.window_size > len(values):
                warnings.warn('Specified window size is bigger than series length... defaulting to file level smoothing.')
            medians = round(values.median()) #sp.mode(values)[0][0]
            values.values[:] = medians

        else:
            # Create windows
            windows = values.rolling(window=self.window_size)
            medians = windows.apply(lambda x: round(x.median()))
            # For each mode change the original value if threshold for percentage is met
            for i, window_median in enumerate(medians):
                if not np.isnan(window_median):
                    values[i] = int(window_median)

        return values
    
    def mean(self, values:pd.Series):
        if self.window_size is None or self.window_size == 0 or self.window_size >= len(values):
            if self.window_size is not None and self.window_size > len(values):
                warnings.warn('Specified window size is bigger than series length... defaulting to file level smoothing.')
            means = round(values.mean()) #sp.mode(values)[0][0]
            values.values[:] = means

        else:
            # Create windows
            windows = values.rolling(window=self.window_size)
            means = windows.apply(lambda x: round(x.mean()))
            # For each mode change the original value if threshold for percentage is met
            for i, window_mean in enumerate(means):
                if not np.isnan(window_mean):
                    values[i] = int(window_mean)

        return values
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--prediction_idx', type=int, default=0)
    parser.add_argument('--group_idx', type=int, default=1)
    parser.add_argument('--mode_threshold', type=float, default=0.75)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--save_group_smoothing', action='store_true')
    parser.add_argument('--save_sample_level_smoothing', action='store_true')
    parser.add_argument('--smoothing_type', type=str, default='mode')

    args = parser.parse_args()

    smooth_csv = Smoothing(args.csv, prediction_idx=args.prediction_idx, group_idx=args.group_idx, threshold=args.threshold, window_size=args.window_size, smoothing_type=args.smoothing_type)

    if args.save_group_smoothing:
        smooth_csv.save_group_csv()
    if args.save_sample_level_smoothing:
        smooth_csv.save_sample_level_csv()