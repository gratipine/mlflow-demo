import pandas as pd

def add_body_flag(dt):
    out = dt.copy()
    out["body_present"] = ~pd.isna(out["body"])
    return out

def add_date_difference_from_start(dt, date_column="dates"):
    out = dt.copy()
    min_date = dt[date_column].min()
    out["diff_from_start"] = out[date_column] - min_date
    return out