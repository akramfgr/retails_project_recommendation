import streamlit as st
import pandas as pd
import sqlite3
from recommendation_engine import RecommendationEngine
from rfm_metrics import RFMMetrics  # Import the RFMMetrics class
from db_utils import load_data, insert_transaction  # Using the refactored database utilities

# Initialize the recommendation engine
@st.cache_resource
def initialize_engine():
    data = load_data()
    engine = RecommendationEngine()
    engine.fit(data[['StockCode', 'Description', 'UnitPrice']])  # Use only relevant columns for recommendations
    return engine

# Compute RFM metrics and predict customer profiles
@st.cache_resource
def calculate_rfm_and_profiles():
    data = load_data()
    data['InvoiceDate'] = pd.to_datetime(
        data['InvoiceDate'],
        infer_datetime_format=True,
        dayfirst=True,  # Adjust if dates are day-first
        errors='coerce'  # Set invalid dates to NaT
    )
    # Drop rows with invalid dates
    data = data.dropna(subset=['InvoiceDate'])

    rfm_calculator = RFMMetrics()
    rfm_data = rfm_calculator.calculate_rfm(data)
    rfm_with_profiles = rfm_calculator.predict_customer_profile(rfm_data)
    return rfm_with_profiles

# Streamlit app
st.set_page_config(layout="wide", page_title="Customer Insights and Recommendation Engine")
st.title("Customer Insights and Recommendation Engine")

# Form Section
st.subheader("Input Customer Details and Purchases")
st.markdown("Please provide the details of your recent purchase.")

# Form Layout
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        customer_id = st.number_input("Customer ID:", min_value=1, step=1)
        invoice_no = st.text_input("Invoice Number:")
    with col2:
        stock_code = st.text_input("Stock Code:")
        description = st.text_input("Description:")

    col3, col4 = st.columns(2)
    with col3:
        quantity = st.number_input("Quantity:", min_value=1, step=1)
        unit_price = st.number_input("Unit Price (€):", min_value=0.01, step=0.01)
    with col4:
        invoice_date = st.date_input("Invoice Date:")
        total_price = unit_price * quantity

    submitted = st.form_submit_button("Submit")

# Process User Input
if submitted:
    with st.spinner("Processing your input..."):
        # Insert transaction into the database
        insert_transaction(
            invoice_no, stock_code, description, quantity, invoice_date, unit_price, customer_id, total_price
        )

        # Recompute RFM and Recommendations
        rfm_data = calculate_rfm_and_profiles()
        customer_rfm = rfm_data[rfm_data['CustomerID'] == customer_id]
        engine = initialize_engine()
        recommendations = engine.get_recommendations([stock_code])

    # Display Results
    st.success("Transaction successfully added and processed!")

    # Two-column layout for results
    col_left, col_right = st.columns(2)

    # Left: RFM Metrics and Customer Profile
    # Left: RFM Metrics and Customer Profile
    # Left: RFM Metrics and Customer Profile
    with col_left:
        st.subheader("Customer RFM Metrics and Profile")
        if not customer_rfm.empty:
            st.table(customer_rfm[['Recency', 'Frequency', 'Monetary', 'Profile']])
            profile = customer_rfm['Profile'].values[0]
            st.info(f"Predicted Customer Profile: **{profile}**")
        else:
            #st.warning("No RFM metrics available for this customer.")
            #st.info("Calculating RFM metrics for the new customer based on form inputs...")

            # Dynamically calculate RFM metrics for the new customer
            rfm_calculator = RFMMetrics()
            new_customer_rfm = rfm_calculator.calculate_rfm_for_new_customer(
                unit_price=unit_price,
                quantity=quantity,
                invoice_date=invoice_date
            )

            # Convert the new customer RFM metrics to a DataFrame for tabular display
            new_customer_rfm_df = pd.DataFrame([new_customer_rfm])

            # Display calculated RFM metrics as a table
            st.table(new_customer_rfm_df)

            # Display default profile information
            st.info("This customer is categorized as 'New' since this is their first transaction.")

    # Right: Recommendations
    with col_right:
        st.subheader("Top 5 Recommendations")
        if not recommendations.empty:
            st.table(recommendations.head(5))
        else:
            st.warning("No recommendations available.")

# Footer
st.markdown("---")
st.caption("© 2024 Customer Insights and Recommendation Engine")