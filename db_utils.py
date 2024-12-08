import sqlite3
import pandas as pd

DB_PATH = 'retail_data.db'

def connect_db():
    """Establish a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def load_data():
    """Load transaction data from the database."""
    conn = connect_db()
    query = """
        SELECT InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, TotalPrice 
        FROM transactions
    """
    try:
        data = pd.read_sql(query, conn)
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
        data = data.dropna(subset=['InvoiceDate'])  # Drop rows with invalid dates
        data['Country'] = data['Country'].fillna("Unknown")
    finally:
        conn.close()
    return data

def insert_transactions(transactions):
    """Insert multiple transactions into the database."""
    conn = connect_db()
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO transactions (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, TotalPrice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        print("Transactions to insert:", transactions)  # Debugging output
        cursor.executemany(query, transactions)
        conn.commit()
    except Exception as e:
        print(f"Unexpected error: {e}")  # Broader exception handling
    finally:
        conn.close()

