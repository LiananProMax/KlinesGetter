# app/tests/test_kline_data_store.py

import unittest
import pandas as pd
from datetime import datetime, timezone, timedelta

# Adjust the import path based on how you run your tests.
# If running from the project root (e.g., python -m unittest discover app/tests),
# this relative import should work.
from app.data_handling.kline_data_store import KlineDataStore

class TestKlineDataStore(unittest.TestCase):

    def setUp(self):
        """Set up a new KlineDataStore for each test."""
        self.base_interval = "1m"
        self.agg_interval = "3m"
        self.display_count = 10 # For memory management testing
        self.store = KlineDataStore(
            base_interval_str=self.base_interval,
            agg_interval_str=self.agg_interval,
            historical_candles_to_display_count=self.display_count
        )
        # Suppress structlog output during tests if it's too noisy
        # You might need a more sophisticated way if structlog is heavily used and configured early
        # For now, we'll proceed. If logs appear, you can configure structlog for tests.
        # import structlog
        # structlog.configure(processors=[]) # Simplistic suppression

    def _create_kline_dict(self, timestamp_dt, open_p, high_p, low_p, close_p, volume_v):
        return {
            'timestamp': timestamp_dt,
            'open': float(open_p),
            'high': float(high_p),
            'low': float(low_p),
            'close': float(close_p),
            'volume': float(volume_v),
            'quote_volume': float(volume_v * close_p) # Approximate quote_volume
        }

    def test_initialization(self):
        """Test if the KlineDataStore initializes with an empty DataFrame."""
        df = self.store.get_klines_df()
        self.assertTrue(df.empty, "DataFrame should be empty upon initialization.")
        self.assertEqual(list(df.columns), ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
        self.assertEqual(self.store.get_base_interval_str(), self.base_interval)
        self.assertEqual(self.store.get_agg_interval_str(), self.agg_interval)
        self.assertEqual(self.store.get_historical_candles_to_display_count(), self.display_count)


    def test_add_single_kline(self):
        """Test adding a single K-line dictionary."""
        dt = pd.to_datetime("2023-01-01 00:00:00", utc=True)
        kline = self._create_kline_dict(dt, 100, 110, 90, 105, 10)
        self.store.add(kline) # Uses the add method which handles single or list

        df = self.store.get_klines_df()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['open'], 100)
        self.assertEqual(df.iloc[0]['timestamp'], dt)

    def test_add_multiple_klines(self):
        """Test adding a list of K-line dictionaries."""
        dt1 = pd.to_datetime("2023-01-01 00:00:00", utc=True)
        dt2 = pd.to_datetime("2023-01-01 00:01:00", utc=True)
        klines = [
            self._create_kline_dict(dt1, 100, 110, 90, 105, 10),
            self._create_kline_dict(dt2, 105, 115, 95, 110, 20)
        ]
        self.store.add(klines) # Uses the add method

        df = self.store.get_klines_df()
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]['close'], 110)

    def test_add_klines_sorting_and_deduplication(self):
        """Test that K-lines are sorted and duplicates (by timestamp) are handled (last one wins)."""
        dt1 = pd.to_datetime("2023-01-01 00:00:00", utc=True)
        dt2 = pd.to_datetime("2023-01-01 00:01:00", utc=True)

        klines_initial = [
            self._create_kline_dict(dt2, 200, 210, 190, 205, 30), # dt2 first
            self._create_kline_dict(dt1, 100, 110, 90, 105, 10),  # dt1 second
        ]
        self.store.add_klines(klines_initial)
        df = self.store.get_klines_df()
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['timestamp'], dt1, "Should be sorted by timestamp.")
        self.assertEqual(df.iloc[1]['timestamp'], dt2)

        # Add a K-line with the same timestamp as dt1, but different data (last one should win)
        kline_duplicate_ts = self._create_kline_dict(dt1, 150, 160, 140, 155, 50)
        self.store.add_single_kline(kline_duplicate_ts)

        df_after_duplicate = self.store.get_klines_df()
        self.assertEqual(len(df_after_duplicate), 2, "Length should remain 2 after duplicate timestamp.")
        # Check if the kline for dt1 was updated
        kline_dt1_updated = df_after_duplicate[df_after_duplicate['timestamp'] == dt1].iloc[0]
        self.assertEqual(kline_dt1_updated['open'], 150)
        self.assertEqual(kline_dt1_updated['close'], 155)

    def test_get_klines_df_returns_copy(self):
        """Test that get_klines_df returns a copy, not a reference to the internal DataFrame."""
        dt = pd.to_datetime("2023-01-01 00:00:00", utc=True)
        kline = self._create_kline_dict(dt, 100, 110, 90, 105, 10)
        self.store.add_single_kline(kline)

        df1 = self.store.get_klines_df()
        # Try to modify the returned DataFrame
        # If it's not a copy, this will affect the internal DataFrame of the store.
        df1.loc[0, 'open'] = 999

        df2 = self.store.get_klines_df()
        self.assertNotEqual(df2.iloc[0]['open'], 999, "get_klines_df should return a copy.")
        self.assertEqual(df2.iloc[0]['open'], 100)

    def test_aggregation(self):
        """Test the get_aggregated method."""
        # Base interval "1m", Agg interval "3m"
        # Add 3 klines for one 3-minute aggregation, and 1 for the next partial one
        klines = [
            self._create_kline_dict(pd.to_datetime("2023-01-01 00:00:00", utc=True), 10, 15, 5,  12, 100),
            self._create_kline_dict(pd.to_datetime("2023-01-01 00:01:00", utc=True), 12, 18, 10, 16, 150),
            self._create_kline_dict(pd.to_datetime("2023-01-01 00:02:00", utc=True), 16, 20, 14, 19, 120),
            self._create_kline_dict(pd.to_datetime("2023-01-01 00:03:00", utc=True), 19, 25, 17, 22, 200), # Start of next 3m candle
        ]
        self.store.add_klines(klines)

        df_agg = self.store.get_aggregated(self.agg_interval)

        self.assertFalse(df_agg.empty)
        self.assertEqual(len(df_agg), 2) # Should produce two 3-minute candles

        # Check first aggregated candle (00:00:00 to 00:02:59)
        agg1 = df_agg.iloc[0]
        self.assertEqual(agg1['timestamp'], pd.to_datetime("2023-01-01 00:00:00", utc=True))
        self.assertEqual(agg1['open'], 10)   # Open of the first 1m kline
        self.assertEqual(agg1['high'], 20)   # Max high of the three 1m klines
        self.assertEqual(agg1['low'], 5)    # Min low of the three 1m klines
        self.assertEqual(agg1['close'], 19)  # Close of the third 1m kline
        self.assertEqual(agg1['volume'], 100 + 150 + 120) # Sum of volumes

        # Check second aggregated candle (00:03:00 to 00:05:59) - will be partial
        agg2 = df_agg.iloc[1]
        self.assertEqual(agg2['timestamp'], pd.to_datetime("2023-01-01 00:03:00", utc=True))
        self.assertEqual(agg2['open'], 19)
        self.assertEqual(agg2['high'], 25)
        self.assertEqual(agg2['low'], 17)
        self.assertEqual(agg2['close'], 22)
        self.assertEqual(agg2['volume'], 200)

    def test_memory_management(self):
        """Test that _manage_memory trims the DataFrame."""
        # base_interval "1m", agg_interval "3m", display_count 10
        # base_intervals_per_agg = 3 / 1 = 3
        # max_base_rows_to_keep = (10 display_count + 20 buffer) * 3 intervals_per_agg
        #                         = 30 * 3 = 90
        max_base_rows_to_keep = (self.display_count + 20) * (pd.Timedelta(self.agg_interval).total_seconds() / pd.Timedelta(self.base_interval).total_seconds())
        self.assertEqual(max_base_rows_to_keep, 90)


        start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        num_klines_to_add = int(max_base_rows_to_keep + 50) # Add more than should be kept

        klines = []
        for i in range(num_klines_to_add):
            ts = start_time + timedelta(minutes=i)
            klines.append(self._create_kline_dict(ts, 100 + i, 110 + i, 90 + i, 105 + i, 10))

        self.store.add_klines(klines)
        df_after_add = self.store.get_klines_df()

        # The _manage_memory method is called internally by add_klines
        self.assertEqual(len(df_after_add), max_base_rows_to_keep,
                         f"DataFrame should be trimmed to {max_base_rows_to_keep} rows.")
        
        # Ensure the latest klines are kept
        expected_last_timestamp = start_time + timedelta(minutes=(num_klines_to_add - 1))
        actual_last_timestamp = df_after_add['timestamp'].iloc[-1]
        self.assertEqual(actual_last_timestamp, expected_last_timestamp)

        expected_first_kept_timestamp = start_time + timedelta(minutes=(num_klines_to_add - int(max_base_rows_to_keep)))
        actual_first_kept_timestamp = df_after_add['timestamp'].iloc[0]
        self.assertEqual(actual_first_kept_timestamp, expected_first_kept_timestamp)


if __name__ == '__main__':
    unittest.main()