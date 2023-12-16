#!/usr/bin/python3

import pandas as pd
import sys


# Load the two CSV files into pandas DataFrames
df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

# Remove the first column from df2
df2 = df2.iloc[:, 1:]

# Concatenate the two DataFrames row-wise
result_df = pd.concat([df1, df2], axis=1)

# Resetting the index of the resulting DataFrame
result_df.reset_index(drop=True, inplace=True)

# Save the concatenated DataFrame to a new CSV file
result_df.to_csv('reg_ecfp_fingerprints.csv', index=False)

