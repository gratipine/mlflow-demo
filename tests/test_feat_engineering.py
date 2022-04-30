from unittest import TestCase
import pandas as pd

import src.feat_engineering as feat


class FeatureEngineeringTests(TestCase):
    def test_date_diff_addition(self):
        dates = pd.to_datetime(
            ["2012-01-04", "2012-01-01", "2012-01-03"],
             format="%Y-%m-%d")
        
        input_dt = pd.DataFrame({
            "index": [1, 2, 3],
            "dates": dates
        })

        expected = pd.DataFrame({
            "index": [1, 2, 3],
            "dates": dates,
            "diff_from_start": [3, 0, 2]
        })

        expected["diff_from_start"] = pd.to_timedelta(expected["diff_from_start"], unit="days")
        result = feat.add_date_difference_from_start(input_dt)
        pd.testing.assert_frame_equal(result, expected)