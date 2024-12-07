import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the existing dataset from SQLite
def load_data():
    conn = sqlite3.connect('retail_data.db')
    query = "SELECT * FROM transactions"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Compute RFM metrics
def compute_rfm(customer_id, updated_data):
    # Process InvoiceDate
    updated_data['InvoiceDate'] = pd.to_datetime(updated_data['InvoiceDate'])
    reference_date = updated_data['InvoiceDate'].max()

    # Compute RFM metrics
    rfm = updated_data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'count',  # Frequency
        'TotalPrice': 'sum'    # Monetary
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Get RFM for the specified customer
    if customer_id in rfm.index:
        return rfm.loc[customer_id].to_dict()
    else:
        return {"error": "Customer not found"}

# Recommendation engine function
# Recommendation engine function
def recommend_items(stock_codes, data):
    # Create customer-item matrix
    customer_item_matrix = data.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        fill_value=0
    )

    # Compute item-item similarity
    item_similarity = cosine_similarity(customer_item_matrix.T)
    item_sim_df = pd.DataFrame(
        item_similarity,
        index=customer_item_matrix.columns,
        columns=customer_item_matrix.columns
    )

    # Generate recommendations
    recommendations = []
    for stock_code in stock_codes:
        if stock_code in item_sim_df.index:
            similar_items = item_sim_df.loc[stock_code].sort_values(ascending=False).head(5).index.tolist()
            recommendations.extend(similar_items)

    # Remove duplicates and get unique recommendations
    unique_recommendations = list(set(recommendations) - set(stock_codes))

    # Get descriptions for recommended items
    recommended_items = data[data['StockCode'].isin(unique_recommendations)][['StockCode', 'Description']].drop_duplicates()

    return recommended_items


# Streamlit App
st.title("")

# Load existing dataset
existing_data = load_data()

# Input Form
st.header("Input Customer Details and Purchases")
with st.form("input_form"):
    customer_id = st.number_input("Customer ID:", min_value=1, step=1)
    invoice_no = st.text_input("Invoice Number:")
    stock_code = st.text_input("Stock Code (comma-separated for multiple items):")
    description = st.text_area("Description:")
    quantity = st.number_input("Quantity:", min_value=1, step=1)
    unit_price = st.number_input("Unit Price:", min_value=0.01, step=0.01, format="%.2f")
    invoice_date = st.date_input("Invoice Date:")
    submitted = st.form_submit_button("Submit")

# Process Input and Update Dataset
if submitted:
    # Parse stock codes as strings
    stock_codes = [code.strip() for code in stock_code.split(",")]

    # Prepare input data
    new_data = pd.DataFrame([{
        "CustomerID": customer_id,
        "InvoiceNo": invoice_no,
        "StockCode": stock_code,
        "Description": description,
        "Quantity": quantity,
        "InvoiceDate": invoice_date,
        "UnitPrice": unit_price,
        "TotalPrice": quantity * unit_price
    }])

    # Combine with existing data
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Compute RFM
    rfm_result = compute_rfm(customer_id, updated_data)

    # Show RFM metrics
    if "error" in rfm_result:
        st.error(rfm_result["error"])
    else:
        st.write("### RFM Metrics")
        st.write(f"**Recency:** {rfm_result['Recency']} days")
        st.write(f"**Frequency:** {rfm_result['Frequency']} transactions")
        st.write(f"**Monetary:** ${rfm_result['Monetary']:.2f}")

    # Generate Recommendations
    recommendations = recommend_items(stock_codes, updated_data)
    st.write("### Recommended Items")
    if not recommendations.empty:
        st.table(recommendations)
    else:
        st.write("No recommendations available.")


# Footer
st.sidebar.info("Customer Insights and Recommendation Engine")
