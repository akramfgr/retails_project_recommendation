import streamlit as st
import pandas as pd
import sqlite3
from recommendation_engine import RecommendationEngine
from rfm_metrics import RFMMetrics
from db_utils import load_data, insert_transactions

# Initialize the recommendation engine
@st.cache_resource
def initialize_engine():
    data = load_data()
    engine = RecommendationEngine()
    engine.fit(data[['StockCode', 'Description', 'UnitPrice']])
    return engine

# Compute RFM metrics and predict customer profiles
@st.cache_resource
def calculate_rfm_and_profiles():
    data = load_data()
    data['InvoiceDate'] = pd.to_datetime(
        data['InvoiceDate'],
        infer_datetime_format=True,
        dayfirst=True,
        errors='coerce'
    )
    data = data.dropna(subset=['InvoiceDate'])
    rfm_calculator = RFMMetrics()
    rfm_data = rfm_calculator.calculate_rfm(data)
    rfm_with_profiles = rfm_calculator.predict_customer_profile(rfm_data)
    return rfm_with_profiles

# Streamlit app setup
st.set_page_config(layout="wide", page_title="Customer Insights and Recommendation Engine")
st.title("Customer Insights and Recommendation Engine")
st.subheader("Input Customer Details and Purchases")
st.markdown("Please provide the details of your recent purchase.")

# Form for inputting the number of items
num_items = st.number_input('Number of items', min_value=1, value=1, step=1, key='num_items')
if 'item_data' not in st.session_state or len(st.session_state.item_data) != num_items:
    st.session_state.item_data = [{'stock_code': '', 'description': '', 'quantity': 1, 'unit_price': 0.01} for _ in range(num_items)]

# Form for inputting details for each item
with st.form("input_form"):
    invoice_no = st.text_input("Invoice Number:")
    customer_id = st.number_input("Customer ID:", min_value=1, step=1)
    invoice_date = st.date_input("Invoice Date:")

    # Dynamically generate input fields for each item
    for i in range(num_items):
        with st.container():
            st.write(f"Item {i+1}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.session_state.item_data[i]['stock_code'] = st.text_input(f"Stock Code {i+1}", key=f"stock_{i}")
            with col2:
                st.session_state.item_data[i]['description'] = st.text_input(f"Description {i+1}", key=f"desc_{i}")
            with col3:
                st.session_state.item_data[i]['quantity'] = st.number_input(f"Quantity {i+1}", min_value=1, step=1, key=f"qty_{i}")
            with col4:
                st.session_state.item_data[i]['unit_price'] = st.number_input(f"Unit Price (€) {i+1}", min_value=0.01, step=0.01, key=f"price_{i}")

    submitted = st.form_submit_button("Submit")

if submitted:
    # Process user input
    with st.spinner("Processing your input..."):
        transactions = [
            (
                invoice_no,
                item['stock_code'],
                item['description'],
                item['quantity'],
                invoice_date,
                item['unit_price'],
                customer_id,
                'Unknown',  # Assuming 'Unknown' for Country
                item['unit_price'] * item['quantity']
            ) for item in st.session_state.item_data
        ]
        insert_transactions(transactions)
        st.success("Transaction successfully added and processed!")

        # Recompute RFM and Recommendations
        rfm_data = calculate_rfm_and_profiles()
        customer_rfm = rfm_data[rfm_data['CustomerID'] == customer_id]

        # Initialize the recommendation engine and get recommendations
        engine = initialize_engine()
        recommendations = engine.get_recommendations([item['stock_code'] for item in st.session_state.item_data])

        # Define columns for layout
        col_left, col_right = st.columns(2)

        if customer_rfm.empty:
            # No existing RFM data; compute dynamically for the new customer
            rfm_calculator = RFMMetrics()
            new_customer_rfm = rfm_calculator.calculate_rfm_for_new_customer(
                unit_price=st.session_state.item_data[-1]['unit_price'],
                quantity=st.session_state.item_data[-1]['quantity'],
                invoice_date=invoice_date
            )
            new_customer_rfm_df = pd.DataFrame([new_customer_rfm])
            profile = "New"

            with col_left:
                st.subheader("Customer RFM Metrics and Profile")
                st.table(new_customer_rfm_df[['Recency', 'Frequency', 'Monetary']])
                st.info(f"Predicted Customer Profile: **{profile}**")
        else:
            with col_left:
                st.subheader("Customer RFM Metrics and Profile")
                st.table(customer_rfm[['Recency', 'Frequency', 'Monetary', 'Profile']])
                profile = customer_rfm['Profile'].values[0]
                st.info(f"Predicted Customer Profile: **{profile}**")

        with col_right:
            st.subheader("Top 5 Recommendations")
            if recommendations.empty:
                st.warning("No recommendations available.")
            else:
                st.table(recommendations.head(5))



# Footer
st.markdown("---")
st.caption("© 2024 Customer Insights and Recommendation Engine")
