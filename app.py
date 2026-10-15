import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD MODEL AND HISTORICAL DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("customer_segmentation_model.pkl")


artifacts = load_model()

scaler = artifacts["scaler"]
kmeans = artifacts["kmeans_model"]
features = artifacts["features"]

@st.cache_data
def load_segment_summary():
    return pd.read_csv("segment_summary.csv")


segment_overview = load_segment_summary()

# --------------------------------------------------
# 3. DASHBOARD HEADER
# --------------------------------------------------
st.title("🛍️ Customer Segmentation System")

st.write(
    "Use customer Recency, Frequency and Monetary values "
    "to identify their customer segment."
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📊 Business Overview",
    "👥 Segment Analysis",
    "🔮 Customer Prediction"
])

# ==================================================
# TAB 1: BUSINESS OVERVIEW
# ==================================================
with tab1:
    st.header("Customer Segmentation Overview")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{segment_overview['Customers'].sum():,}"
    )

    col2.metric(
        "Customer Segments",
        f"{segment_overview['Cluster'].nunique()}"
    )

    col3.metric(
        "Historical Revenue",
        f"£{segment_overview['Revenue'].sum():,.2f}"
    )

    st.divider()

    # Customer count chart
    customer_chart = px.bar(
        segment_overview,
        x="Segment",
        y="Customers",
        color="Segment",
        title="Customers per Segment",
        labels={"Customers": "Number of Customers"}
    )

    st.plotly_chart(
        customer_chart,
        use_container_width=True
    )

    # Revenue contribution chart
    revenue_chart = px.pie(
        segment_overview,
        names="Segment",
        values="Revenue",
        title="Historical Revenue Contribution",
        hole=0.4
    )

    st.plotly_chart(
        revenue_chart,
        use_container_width=True
    )

# ==================================================
# TAB 2: SEGMENT ANALYSIS
# ==================================================
with tab2:
    st.header("Customer Segment Analysis")

    # Segment filter
    selected_segment = st.selectbox(
        "Filter by customer segment",
        options=[
            "All Segments"
        ] + segment_overview["Segment"].tolist()
    )

    # Filter the segment data
    if selected_segment == "All Segments":
        filtered_overview = segment_overview.copy()
    else:
        filtered_overview = segment_overview[
            segment_overview["Segment"] == selected_segment
        ]

    # Display individual segment metrics
    if selected_segment != "All Segments":
        selected_row = filtered_overview.iloc[0]

        st.subheader(selected_row["Segment"])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Customers",
            f"{int(selected_row['Customers']):,}"
        )

        col2.metric(
            "Revenue",
            f"£{selected_row['Revenue']:,.2f}"
        )

        col3.metric(
            "Revenue Share",
            f"{selected_row['Revenue Share (%)']:.2f}%"
        )

    # Format a separate copy for display
    display_table = filtered_overview.copy()

    display_table["Revenue"] = display_table["Revenue"].map(
        lambda amount: f"£{amount:,.2f}"
    )

    st.subheader("Segment Performance Table")

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    # Download segment report
    csv_data = filtered_overview.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Segment Report (CSV)",
        data=csv_data,
        file_name="customer_segment_report.csv",
        mime="text/csv"
    )

# ==================================================
# TAB 3: CUSTOMER PREDICTION
# ==================================================
with tab3:
    st.header("Enter Customer Information")

    # Customer input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (days since last purchase)",
            min_value=1,
            value=30
        )

    with col2:
        frequency = st.number_input(
            "Frequency (number of invoices)",
            min_value=1,
            value=3
        )

    with col3:
        monetary = st.number_input(
            "Monetary (total spending in £)",
            min_value=0.01,
            value=500.0,
            step=50.0
        )

    # Predict customer segment
    if st.button("Predict Customer Segment", type="primary"):

        # ------------------------------------------
        # INPUT VALIDATION WARNINGS
        # ------------------------------------------
        unusual_values = []

        if recency > 374:
            unusual_values.append(
                "Recency exceeds the observed maximum of 374 days."
            )

        if frequency > 209:
            unusual_values.append(
                "Frequency exceeds the observed maximum of 209 invoices."
            )

        if monetary > 280206.02:
            unusual_values.append(
                "Monetary exceeds the observed maximum of £280,206.02."
            )

        if unusual_values:
            st.warning(
                "These inputs are outside the historical training-data "
                "ranges. Interpret the prediction cautiously."
            )

            for message in unusual_values:
                st.write(f"- {message}")

        # ------------------------------------------
        # CUSTOMER INPUT SUMMARY
        # ------------------------------------------
        st.subheader("Customer Input Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        summary_col1.metric("Recency", f"{recency} days")
        summary_col2.metric("Frequency", f"{frequency} invoices")
        summary_col3.metric("Monetary", f"£{monetary:,.2f}")

        # ------------------------------------------
        # PREPARE CUSTOMER DATA
        # ------------------------------------------
        customer = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=features
        )

        # Apply the same transformation used during training
        customer_log = np.log1p(customer)
        customer_scaled = scaler.transform(customer_log)

        # Preserve feature names for K-means
        customer_scaled = pd.DataFrame(
            customer_scaled,
            columns=features
        )

        # Predict cluster
        cluster = int(kmeans.predict(customer_scaled)[0])

        # ------------------------------------------
        # BUSINESS INTERPRETATION
        # ------------------------------------------
        segment_details = {
            0: {
                "name": "High-Value, Loyal Customers",
                "description": (
                    "These customers purchase recently, buy frequently, "
                    "and have high historical spending."
                ),
                "action": (
                    "Prioritise retention, loyalty rewards, exclusive "
                    "offers, and personalised recommendations."
                )
            },
            1: {
                "name": "Lapsed, Low-Value Customers",
                "description": (
                    "These customers have not purchased recently and "
                    "typically have low purchase frequency and spending."
                ),
                "action": (
                    "Test targeted win-back campaigns and modest "
                    "incentives. Measure whether reactivation is profitable."
                )
            },
            2: {
                "name": "Recent, Occasional Customers",
                "description": (
                    "These customers purchased recently but have "
                    "relatively low purchase frequency and spending."
                ),
                "action": (
                    "Encourage repeat purchases with relevant product "
                    "recommendations and complementary-product offers."
                )
            },
            3: {
                "name": "Established, Moderate-Value Customers",
                "description": (
                    "These customers have moderate purchase recency, "
                    "frequency, and historical spending."
                ),
                "action": (
                    "Test cross-selling, useful product bundles, and "
                    "personalised recommendations to grow customer value."
                )
            }
        }

        segment = segment_details[cluster]

        # ------------------------------------------
        # PREDICTION REPORT
        # ------------------------------------------
        prediction_report = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "Predicted Cluster": cluster,
            "Customer Segment": segment["name"],
            "Recommended Action": segment["action"]
        }])

        prediction_csv = prediction_report.to_csv(
            index=False
        ).encode("utf-8")

        # ------------------------------------------
        # DISPLAY PREDICTION
        # ------------------------------------------
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Cluster",
                f"Cluster {cluster}"
            )

        with result_col2:
            st.success(segment["name"])

        st.write(
            f"**Customer behaviour:** {segment['description']}"
        )

        st.subheader("Recommended Business Action")
        st.write(segment["action"])

        # Download individual prediction
        st.download_button(
            label="Download Customer Prediction (CSV)",
            data=prediction_csv,
            file_name="customer_prediction.csv",
            mime="text/csv"
        )

        st.caption(
            "Recommendations are based on historical segment profiles "
            "and should be tested before use in marketing campaigns."
        )
