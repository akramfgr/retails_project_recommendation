import sqlite3
import pandas as pd

# Load the cleaned dataset
file_path = 'cleaned_dataset.csv'  # Replace with your cleaned dataset file
data = pd.read_csv(file_path)

# Connect to SQLite and save the data
conn = sqlite3.connect('retail_data.db')
data.to_sql('transactions', conn, if_exists='replace', index=False)
conn.close()

print("Data successfully saved to SQLite database.")
