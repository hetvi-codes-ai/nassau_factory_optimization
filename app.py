import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Nassau Candy Optimization",
    layout="wide"
)

st.title("🍬 Nassau Candy Factory Optimization System")

# ==========================
# LOAD FILES
# ==========================

recommendations = pd.read_csv(
    "recommendation_outputs/final_recommendations.csv"
)

simulation = pd.read_csv(
    "simulation_outputs/simulation_results.csv"
)

clusters = pd.read_csv(
    "clustering_outputs/clustered_data.csv"
)

# ==========================
# SIDEBAR
# ==========================

page = st.sidebar.selectbox(

    "Select Module",

    [

        "Factory Optimization Simulator",

        "What-If Analysis",

        "Recommendation Dashboard",

        "Risk & Impact Panel"

    ]

)

# =====================================================
# PAGE 1
# =====================================================

if page == "Factory Optimization Simulator":

    st.header("Factory Optimization Simulator")

    products = simulation[
        "Product"
    ].unique()

    selected_product = st.selectbox(

        "Select Product",

        products

    )

    product_data = simulation[
        simulation["Product"] == selected_product
    ]

    st.dataframe(product_data)

    fig = px.bar(

        product_data,

        x="Alternative Factory",

        y="Predicted Lead Time",

        color="Alternative Factory",

        title="Predicted Lead Time by Factory"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# PAGE 2
# =====================================================

elif page == "What-If Analysis":

    st.header("What-If Scenario Analysis")

    products = simulation[
        "Product"
    ].unique()

    selected_product = st.selectbox(

        "Product",

        products

    )

    product_data = simulation[
        simulation["Product"] == selected_product
    ]

    st.dataframe(product_data)

    fig = px.bar(

        product_data,

        x="Alternative Factory",

        y="Lead Time Improvement %",

        color="Alternative Factory",

        title="Lead Time Improvement"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# PAGE 3
# =====================================================

elif page == "Recommendation Dashboard":

    st.header("Recommendation Dashboard")

    st.dataframe(
        recommendations
    )

    fig = px.bar(

        recommendations,

        x="Product",

        y="Lead Time Improvement %",

        color="Recommended Factory",

        title="Recommended Factory Reassignments"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# PAGE 4
# =====================================================

elif page == "Risk & Impact Panel":

    st.header("Risk & Impact Analysis")

    st.dataframe(
        recommendations
    )

    risk_count = recommendations[
        "Risk Score"
    ].value_counts()

    fig = px.pie(

        names=risk_count.index,

        values=risk_count.values,

        title="Risk Distribution"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.scatter(

        recommendations,

        x="Confidence Score",

        y="Lead Time Improvement %",

        color="Risk Score",

        hover_data=["Product"],

        title="Confidence vs Improvement"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )