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
    data = pd.read_sql(query, conn)
    conn.close()
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
    data = data.dropna(subset=['InvoiceDate'])  # Drop rows with invalid dates
    data['Country'] = data['Country'].fillna("Unknown")
    return data

def insert_transaction(invoice_no, stock_code, description, quantity, invoice_date, unit_price, customer_id, total_price, country="Unknown"):
    """Insert a new transaction into the database."""
    conn = connect_db()
    cursor = conn.cursor()
    query = """
        INSERT INTO transactions (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, TotalPrice)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(query, (invoice_no, stock_code, description, quantity, invoice_date, unit_price, customer_id, country, total_price))
    conn.commit()
    conn.close()
