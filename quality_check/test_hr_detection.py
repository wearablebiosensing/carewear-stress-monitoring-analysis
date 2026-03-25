import pandas as pd
import sys
import os

# Add the directory containing heart_rate_check.py to sys.path
sys.path.append(os.path.abspath('.'))

from heart_rate_check import standardize_hr_column

def test_standardize_hr_column():
    # Test 1: Exact match "event"
    df1 = pd.DataFrame({"event": [60, 70, 80], "Timestamp": [1, 2, 3]})
    df1 = standardize_hr_column(df1)
    assert "HeartRate" in df1.columns
    assert list(df1["HeartRate"]) == [60, 70, 80]
    print("Test 1 (exact 'event') passed!")

    # Test 2: Case-insensitive "Event"
    df2 = pd.DataFrame({"Event": [60, 70, 80], "Timestamp": [1, 2, 3]})
    df2 = standardize_hr_column(df2)
    assert "HeartRate" in df2.columns
    assert list(df2["HeartRate"]) == [60, 70, 80]
    print("Test 2 (case-insensitive 'Event') passed!")

    # Test 3: Whitespace " event "
    df3 = pd.DataFrame({"  event  ": [60, 70, 80], "Timestamp": [1, 2, 3]})
    df3 = standardize_hr_column(df3)
    assert "HeartRate" in df3.columns
    assert list(df3["HeartRate"]) == [60, 70, 80]
    print("Test 3 (whitespace ' event ') passed!")

    # Test 4: Fallback "HR_Value"
    df4 = pd.DataFrame({"HR_Value": [60, 70, 80], "Timestamp": [1, 2, 3]})
    df4 = standardize_hr_column(df4)
    assert "HeartRate" in df4.columns
    assert list(df4["HeartRate"]) == [60, 70, 80]
    print("Test 4 (fallback 'HR_Value') passed!")

if __name__ == "__main__":
    try:
        test_standardize_hr_column()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
