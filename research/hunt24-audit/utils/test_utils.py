import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from astropy.table import Table

# Import the functions to test
# Adjust the import path according to your project structure
from decorators import astro_checkpoint
from vot_to_parquet import votable_to_parquet

class TestAstroUtils(unittest.TestCase):

    # --- Tests for astro_checkpoint decorator ---

    def setUp(self):
        # Mock database and instance for decorator tests
        self.db_mock = MagicMock()
        self.db_mock.con = MagicMock()
        self.instance = MagicMock()
        self.instance.db = self.db_mock

    def test_astro_checkpoint_uses_existing_cache(self):
        """Test that the decorator reuses the cache if it exists and force_refresh is False."""
        self.db_mock.con.execute.return_value.fetchone.return_value = [1] # Cache exists
        
        call_count = 0
        @astro_checkpoint("test_table", force_refresh=False)
        def decorated_func(self_inner):
            nonlocal call_count
            call_count += 1
            return "actual_table"

        result = decorated_func(self.instance)
        
        self.assertEqual(result, "test_table")
        self.assertEqual(call_count, 0) # Original function should NOT be called
        self.db_mock.con.execute.assert_called_with(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'test_table'"
        )

    def test_astro_checkpoint_runs_func_when_no_cache(self):
        """Test that the decorator runs the function and creates cache if none exists."""
        self.db_mock.con.execute.return_value.fetchone.return_value = [0] # Cache doesn't exist
        
        @astro_checkpoint("test_table")
        def decorated_func(self_inner):
            return "physical_table"

        result = decorated_func(self.instance)
        
        self.assertEqual(result, "test_table")
        # Verify cache creation logic
        calls = [c[0][0] for c in self.db_mock.con.execute.call_args_list]
        self.assertTrue(any("DROP TABLE IF EXISTS test_table" in s for s in calls))
        self.assertTrue(any("CREATE TABLE test_table AS SELECT * FROM physical_table" in s for s in calls))

    def test_astro_checkpoint_force_refresh_true(self):
        """Test that force_refresh=True triggers execution even if cache exists."""
        self.db_mock.con.execute.return_value.fetchone.return_value = [1] # Cache exists
        
        call_count = 0
        @astro_checkpoint("test_table", force_refresh=True)
        def decorated_func(self_inner):
            nonlocal call_count
            call_count += 1
            return "new_physical_table"

        result = decorated_func(self.instance)
        
        self.assertEqual(result, "test_table")
        self.assertEqual(call_count, 1) # Function SHOULD be called
        calls = [c[0][0] for c in self.db_mock.con.execute.call_args_list]
        self.assertTrue(any("CREATE TABLE test_table AS SELECT * FROM new_physical_table" in s for s in calls))

    # --- Tests for votable_to_parquet ---

    @patch('astropy.table.Table.read')
    @patch('pandas.DataFrame.to_parquet')
    def test_votable_to_parquet_full_flow(self, mock_to_parquet, mock_table_read):
        """Test the full conversion flow including bytes decoding."""
        # Setup mock data with bytes
        data = {
            'id': [1, 2],
            'source_id': [b'123', b'456'], # bytes column
            'label': ['A', 'B'] # string column
        }
        df = pd.DataFrame(data)
        
        mock_table = MagicMock(spec=Table)
        mock_table.to_pandas.return_value = df
        mock_table_read.return_value = mock_table
        
        votable_to_parquet("input.vot", "output.parquet")
        
        # Verify Astropy read call
        mock_table_read.assert_called_once_with("input.vot", format='votable')
        
        # Verify bytes were decoded to strings
        self.assertEqual(df['source_id'][0], '123')
        self.assertIsInstance(df['source_id'][0], str)
        self.assertEqual(df['label'][0], 'A')
        
        # Verify Parquet write call
        mock_to_parquet.assert_called_once_with("output.parquet", engine='pyarrow', index=False)

if __name__ == '__main__':
    unittest.main()
