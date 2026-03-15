# streamlit_dashboard.py
"""
Interactive Streamlit Dashboard for Instacart ML Pipeline

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Instacart ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
        border-bottom: 3px solid #1f77b4;
    }
    h2 {
        color: #ff7f0e;
        margin-top: 30px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load predictions and features"""
    predictions = pd.read_parquet("data/predictions/predictions.parquet")
    features = pd.read_parquet("data/features/feature_store.parquet")
    return predictions, features

# Load data
try:
    predictions, features = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("Make sure you've run the prediction pipeline first!")
    data_loaded = False
    st.stop()

# Sidebar
st.sidebar.title("🛒 Instacart ML Dashboard")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Overview", "📊 Model Performance", "👥 User Analysis", 
     "🛍️ Product Analysis", "🎯 Recommendations", "📈 Business Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.metric("Total Users", f"{predictions['user_id'].nunique():,}")
st.sidebar.metric("Total Products", f"{predictions['product_id'].nunique():,}")
st.sidebar.metric("Total Predictions", f"{len(predictions):,}")

# Main content
if page == "🏠 Overview":
    st.title("🛒 Instacart Multi-Model ML Pipeline")
    st.markdown("### Comprehensive Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Avg Basket Size",
            f"{predictions['basket_model_prediction'].mean():.2f}",
            f"{predictions['basket_model_prediction'].std():.2f} σ"
        )
    
    with col2:
        reorder_rate = predictions['reorder_model_prediction'].mean()
        st.metric(
            "Reorder Rate",
            f"{reorder_rate*100:.1f}%",
            "Classification"
        )
    
    with col3:
        churn_rate = predictions.groupby('user_id')['churn_model_prediction'].first().mean()
        st.metric(
            "Churn Rate",
            f"{churn_rate*100:.1f}%",
            "At Risk" if churn_rate > 0.3 else "Healthy"
        )
    
    with col4:
        st.metric(
            "Unique Users",
            f"{predictions['user_id'].nunique():,}",
            "Active"
        )
    
    with col5:
        st.metric(
            "Unique Products",
            f"{predictions['product_id'].nunique():,}",
            "In Catalog"
        )
    
    st.markdown("---")
    
    # Models overview
    st.subheader("🤖 Trained Models")
    
    models_data = {
        "Model": ["Basket Size", "Reorder Probability", "Churn Prediction", "Next Product", "Recommendation"],
        "Type": ["Regression", "Classification", "Classification", "Ranking", "Ranking"],
        "Status": ["✅ Active", "✅ Active", "✅ Active", "✅ Active", "✅ Active"],
        "Use Case": [
            "Predict cart size for inventory planning",
            "Predict product reorder likelihood",
            "Identify users at risk of churning",
            "Rank products for personalized suggestions",
            "General product recommendations"
        ]
    }
    
    st.dataframe(
        pd.DataFrame(models_data),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution Overview")
        
        # Basket size distribution
        fig = px.histogram(
            predictions,
            x='basket_model_prediction',
            nbins=50,
            title="Basket Size Distribution",
            labels={'basket_model_prediction': 'Predicted Basket Size'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Model Predictions")
        
        # Reorder vs Churn pie
        user_data = predictions.groupby('user_id').agg({
            'reorder_model_prediction': 'mean',
            'churn_model_prediction': 'first'
        })
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type':'pie'}, {'type':'pie'}]],
            subplot_titles=('Reorder Rate', 'Churn Risk')
        )
        
        # Reorder pie
        reorder_counts = [
            len(user_data[user_data['reorder_model_prediction'] <= 0.5]),
            len(user_data[user_data['reorder_model_prediction'] > 0.5])
        ]
        fig.add_trace(
            go.Pie(labels=['Low Reorder', 'High Reorder'], values=reorder_counts,
                   marker_colors=['#ff9999', '#66b3ff']),
            row=1, col=1
        )
        
        # Churn pie
        churn_counts = user_data['churn_model_prediction'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Active', 'At Risk'], 
                   values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                   marker_colors=['#90EE90', '#FF6B6B']),
            row=1, col=2
        )
        
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Model Performance":
    st.title("📊 Model Performance Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🛒 Basket Size", "🔄 Reorder", "⚠️ Churn", "⭐ Next Product", "💡 Recommendations"
    ])
    
    with tab1:
        st.subheader("Basket Size Prediction Model")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Prediction", f"{predictions['basket_model_prediction'].mean():.2f}")
        with col2:
            st.metric("Median Prediction", f"{predictions['basket_model_prediction'].median():.2f}")
        with col3:
            st.metric("Std Deviation", f"{predictions['basket_model_prediction'].std():.2f}")
        
        # Distribution
        fig = px.histogram(
            predictions,
            x='basket_model_prediction',
            nbins=100,
            title="Distribution of Basket Size Predictions",
            labels={'basket_model_prediction': 'Predicted Basket Size'},
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # By user
        user_basket = predictions.groupby('user_id')['basket_model_prediction'].mean().reset_index()
        user_basket.columns = ['user_id', 'avg_basket']
        
        fig = px.histogram(
            user_basket,
            x='avg_basket',
            nbins=50,
            title="Average Basket Size per User",
            labels={'avg_basket': 'Average Basket Size'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top users
        top_users = user_basket.nlargest(20, 'avg_basket')
        fig = px.bar(
            top_users,
            x='user_id',
            y='avg_basket',
            title="Top 20 Users by Average Basket Size",
            labels={'avg_basket': 'Avg Basket Size', 'user_id': 'User ID'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Reorder Probability Model")
        
        reorder_rate = predictions['reorder_model_prediction'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Reorder Rate", f"{reorder_rate*100:.1f}%")
        with col2:
            high_reorder = (predictions['reorder_model_prediction'] > 0.7).sum()
            st.metric("High Confidence Reorders", f"{high_reorder:,}")
        
        # Product reorder analysis
        product_reorder = predictions.groupby('product_id').agg({
            'reorder_model_prediction': ['sum', 'mean', 'count']
        }).reset_index()
        product_reorder.columns = ['product_id', 'total_reorders', 'avg_rate', 'count']
        
        # Top reordered products
        top_products = product_reorder.nlargest(30, 'total_reorders')
        fig = px.bar(
            top_products,
            x='product_id',
            y='total_reorders',
            title="Top 30 Products by Reorder Predictions",
            labels={'total_reorders': 'Total Reorder Predictions'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Reorder rate distribution
        fig = px.scatter(
            product_reorder,
            x='count',
            y='avg_rate',
            title="Product Reorder Rate vs Popularity",
            labels={'count': 'Number of Users', 'avg_rate': 'Average Reorder Rate'},
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Churn Prediction Model")
        
        user_churn = predictions.groupby('user_id')['churn_model_prediction'].first()
        churn_rate = user_churn.mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Rate", f"{churn_rate*100:.1f}%")
        with col2:
            st.metric("Active Users", f"{(user_churn == 0).sum():,}")
        with col3:
            st.metric("At-Risk Users", f"{(user_churn == 1).sum():,}")
        
        # Churn distribution
        churn_counts = user_churn.value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=['Active Users', 'At-Risk Users'],
                values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                hole=0.4,
                marker_colors=['#90EE90', '#FF6B6B']
            )
        ])
        fig.update_layout(title="User Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature comparison
        if 'user_total_orders' in predictions.columns:
            churned = predictions[predictions['churn_model_prediction'] == 1]
            active = predictions[predictions['churn_model_prediction'] == 0]
            
            comparison = pd.DataFrame({
                'Metric': ['Avg Orders', 'Avg Products', 'Avg Basket'],
                'Active Users': [
                    active.groupby('user_id')['user_total_orders'].first().mean(),
                    active.groupby('user_id')['user_unique_products'].first().mean(),
                    active['basket_model_prediction'].mean()
                ],
                'Churned Users': [
                    churned.groupby('user_id')['user_total_orders'].first().mean(),
                    churned.groupby('user_id')['user_unique_products'].first().mean(),
                    churned['basket_model_prediction'].mean()
                ]
            })
            
            fig = go.Figure(data=[
                go.Bar(name='Active Users', x=comparison['Metric'], y=comparison['Active Users']),
                go.Bar(name='Churned Users', x=comparison['Metric'], y=comparison['Churned Users'])
            ])
            fig.update_layout(
                title="Active vs Churned User Behavior",
                barmode='group',
                yaxis_title="Average Value"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Next Product Prediction Model")
        st.info("This ranking model predicts which products a user is most likely to purchase next.")
        
        # Sample user analysis
        sample_user = st.selectbox(
            "Select a user to see their top predicted products:",
            predictions['user_id'].unique()[:100]
        )
        
        user_preds = predictions[predictions['user_id'] == sample_user].nlargest(
            15, 'next_product_model_prediction'
        )
        
        fig = px.bar(
            user_preds,
            x='product_id',
            y='next_product_model_prediction',
            title=f"Top 15 Product Predictions for User {sample_user}",
            labels={'next_product_model_prediction': 'Prediction Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Recommendation System Model")
        
        # Top recommended products overall
        top_recs = predictions.groupby('product_id')['recommendation_model_prediction'].mean().nlargest(30)
        
        fig = px.bar(
            x=top_recs.index,
            y=top_recs.values,
            title="Top 30 Recommended Products (Overall)",
            labels={'x': 'Product ID', 'y': 'Avg Recommendation Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution
        fig = px.histogram(
            predictions,
            x='recommendation_model_prediction',
            nbins=50,
            title="Distribution of Recommendation Scores",
            labels={'recommendation_model_prediction': 'Recommendation Score'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "👥 User Analysis":
    st.title("👥 User Segmentation & Analysis")
    
    # User-level aggregation
    user_summary = predictions.groupby('user_id').agg({
        'basket_model_prediction': 'mean',
        'reorder_model_prediction': 'mean',
        'churn_model_prediction': 'first',
        'product_id': 'count'
    }).reset_index()
    user_summary.columns = ['user_id', 'avg_basket', 'reorder_rate', 'churn_risk', 'total_products']
    
    # Segmentation
    st.subheader("📊 User Segments")
    
    def segment_users(row):
        if row['churn_risk'] == 1:
            return "At Risk"
        elif row['reorder_rate'] > 0.6 and row['avg_basket'] > user_summary['avg_basket'].median():
            return "VIP"
        elif row['reorder_rate'] > 0.6:
            return "Loyal"
        elif row['avg_basket'] > user_summary['avg_basket'].median():
            return "High Value"
        else:
            return "Regular"
    
    user_summary['segment'] = user_summary.apply(segment_users, axis=1)
    
    # Segment distribution
    segment_counts = user_summary['segment'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            segment_counts.reset_index(),
            column_config={
                "index": "Segment",
                "count": "Users"
            },
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="User Segment Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment characteristics
    st.subheader("📈 Segment Characteristics")
    
    segment_stats = user_summary.groupby('segment').agg({
        'avg_basket': 'mean',
        'reorder_rate': 'mean',
        'total_products': 'mean'
    }).round(2)
    
    st.dataframe(segment_stats, use_container_width=True)
    
    # Scatter plot
    fig = px.scatter(
        user_summary,
        x='avg_basket',
        y='reorder_rate',
        color='segment',
        size='total_products',
        title="User Segmentation: Basket Size vs Reorder Rate",
        labels={
            'avg_basket': 'Average Basket Size',
            'reorder_rate': 'Reorder Rate',
            'total_products': 'Total Products'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top users
    st.subheader("🏆 Top Users")
    
    metric_choice = st.selectbox(
        "Rank by:",
        ["Average Basket Size", "Reorder Rate", "Total Products"]
    )
    
    metric_map = {
        "Average Basket Size": "avg_basket",
        "Reorder Rate": "reorder_rate",
        "Total Products": "total_products"
    }
    
    top_users = user_summary.nlargest(20, metric_map[metric_choice])
    
    fig = px.bar(
        top_users,
        x='user_id',
        y=metric_map[metric_choice],
        color='segment',
        title=f"Top 20 Users by {metric_choice}"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "🛍️ Product Analysis":
    st.title("🛍️ Product Performance Analysis")
    
    # Product-level aggregation
    product_summary = predictions.groupby('product_id').agg({
        'basket_model_prediction': 'mean',
        'reorder_model_prediction': ['sum', 'mean'],
        'recommendation_model_prediction': 'mean',
        'user_id': 'count'
    }).reset_index()
    product_summary.columns = [
        'product_id', 'avg_basket_position', 'total_reorders', 
        'reorder_rate', 'rec_score', 'num_users'
    ]
    
    # Top products
    st.subheader("🏆 Top Products")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Products", f"{len(product_summary):,}")
    with col2:
        st.metric("Avg Reorder Rate", f"{product_summary['reorder_rate'].mean()*100:.1f}%")
    with col3:
        st.metric("Avg Users per Product", f"{product_summary['num_users'].mean():.1f}")
    
    # Interactive table
    st.subheader("📊 Product Performance Table")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_users = st.slider("Minimum number of users", 0, 100, 10)
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Total Reorders", "Reorder Rate", "Recommendation Score", "Number of Users"]
        )
    
    sort_map = {
        "Total Reorders": "total_reorders",
        "Reorder Rate": "reorder_rate",
        "Recommendation Score": "rec_score",
        "Number of Users": "num_users"
    }
    
    filtered_products = product_summary[product_summary['num_users'] >= min_users].sort_values(
        sort_map[sort_by], ascending=False
    ).head(50)
    
    st.dataframe(filtered_products, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.subheader("📈 Product Insights")
    
    tab1, tab2, tab3 = st.tabs(["Reorder Analysis", "Popularity", "Recommendations"])
    
    with tab1:
        fig = px.scatter(
            product_summary,
            x='num_users',
            y='reorder_rate',
            size='total_reorders',
            title="Product Popularity vs Reorder Rate",
            labels={
                'num_users': 'Number of Users',
                'reorder_rate': 'Reorder Rate',
                'total_reorders': 'Total Reorders'
            },
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        top_popular = product_summary.nlargest(30, 'num_users')
        fig = px.bar(
            top_popular,
            x='product_id',
            y='num_users',
            title="Top 30 Most Popular Products",
            labels={'num_users': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        top_recommended = product_summary.nlargest(30, 'rec_score')
        fig = px.bar(
            top_recommended,
            x='product_id',
            y='rec_score',
            title="Top 30 Recommended Products",
            labels={'rec_score': 'Recommendation Score'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Recommendations":
    st.title("🎯 Personalized Recommendations")
    
    st.markdown("""
    This section allows you to explore personalized product recommendations for specific users.
    """)
    
    # User selector
    user_id = st.number_input(
        "Enter User ID:",
        min_value=int(predictions['user_id'].min()),
        max_value=int(predictions['user_id'].max()),
        value=int(predictions['user_id'].iloc[0])
    )
    
    if user_id in predictions['user_id'].values:
        user_data = predictions[predictions['user_id'] == user_id]
        
        # User stats
        st.subheader(f"📊 User {user_id} Profile")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Basket Size", f"{user_data['basket_model_prediction'].mean():.2f}")
        with col2:
            st.metric("Reorder Rate", f"{user_data['reorder_model_prediction'].mean()*100:.1f}%")
        with col3:
            churn_status = "⚠️ At Risk" if user_data['churn_model_prediction'].iloc[0] == 1 else "✅ Active"
            st.metric("Status", churn_status)
        with col4:
            st.metric("Products", f"{len(user_data):,}")
        
        # Top recommendations
        st.subheader("⭐ Top 20 Product Recommendations")
        
        top_recs = user_data.nlargest(20, 'recommendation_model_prediction')
        
        fig = px.bar(
            top_recs,
            x='product_id',
            y='recommendation_model_prediction',
            color='reorder_model_prediction',
            title=f"Top Recommendations for User {user_id}",
            labels={
                'recommendation_model_prediction': 'Recommendation Score',
                'reorder_model_prediction': 'Reorder Probability'
            },
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Recommendation Details")
        
        rec_table = top_recs[['product_id', 'recommendation_model_prediction', 
                               'reorder_model_prediction', 'basket_model_prediction']].copy()
        rec_table.columns = ['Product ID', 'Rec Score', 'Reorder Prob', 'Basket Position']
        rec_table = rec_table.round(4)
        
        st.dataframe(rec_table, use_container_width=True, hide_index=True)
        
    else:
        st.warning(f"User {user_id} not found in the dataset.")

elif page == "📈 Business Insights":
    st.title("📈 Business Insights & Actions")
    
    # Calculate key insights
    user_summary = predictions.groupby('user_id').agg({
        'basket_model_prediction': 'mean',
        'reorder_model_prediction': 'mean',
        'churn_model_prediction': 'first'
    })
    
    churned_users = user_summary[user_summary['churn_model_prediction'] == 1]
    high_value_users = user_summary[
        (user_summary['basket_model_prediction'] > user_summary['basket_model_prediction'].quantile(0.75)) &
        (user_summary['reorder_model_prediction'] > 0.6)
    ]
    
    # Key insights
    st.subheader("🎯 Strategic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Market Opportunity
        
        **Customer Retention**
        - 🔴 **{:,} users at risk** of churning ({:.1f}%)
        - 💡 Implement targeted retention campaigns
        - 🎁 Offer personalized discounts to at-risk segment
        
        **Revenue Optimization**
        - 📦 Average basket: **{:.2f} items**
        - 📈 Opportunity to increase through cross-selling
        - 🎯 Target users with below-average baskets
        """.format(
            len(churned_users),
            len(churned_users)/len(user_summary)*100,
            predictions['basket_model_prediction'].mean()
        ))
    
    with col2:
        st.markdown("""
        ### 💎 High-Value Segments
        
        **VIP Users**
        - ⭐ **{:,} high-value users** identified
        - 🛒 High basket size + high reorder rate
        - 💝 Priority support and exclusive offers
        
        **Product Strategy**
        - 🔄 {:.1f}% overall reorder rate
        - 📦 Create "Buy Again" quick-access feature
        - 🎁 Bundle high-reorder products
        """.format(
            len(high_value_users),
            predictions['reorder_model_prediction'].mean()*100
        ))
    
    st.markdown("---")
    
    # Action items
    st.subheader("✅ Recommended Actions")
    
    action1, action2, action3 = st.columns(3)
    
    with action1:
        st.markdown("""
        ### 1️⃣ Retention Campaign
        
        **Target:** {:,} at-risk users
        
        **Actions:**
        - Send personalized "We miss you" emails
        - Offer 15-20% discount on next order
        - Highlight their favorite products
        - Mobile push notifications
        
        **Expected Impact:** 30-40% retention
        """.format(len(churned_users)))
    
    with action2:
        st.markdown("""
        ### 2️⃣ Recommendation Engine
        
        **Deploy personalized recommendations:**
        
        - ✉️ Email: Weekly product suggestions
        - 📱 App: Homepage recommendations
        - 🛒 Cart: "Frequently bought together"
        - 🔔 Push: Restock notifications
        
        **Expected Impact:** +15% basket size
        """)
    
    with action3:
        st.markdown("""
        ### 3️⃣ Product Bundling
        
        **Create smart bundles:**
        
        - 🥗 Weekly meal kits
        - 🧹 Household essentials pack
        - 🍼 Baby care bundle
        - 🐕 Pet supplies combo
        
        **Expected Impact:** +20% AOV
        """)
    
    st.markdown("---")
    
    # Download reports
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        churned_csv = churned_users.to_csv(index=True)
        st.download_button(
            label="📄 Download Churned Users",
            data=churned_csv,
            file_name="churned_users.csv",
            mime="text/csv"
        )
    
    with col2:
        high_value_csv = high_value_users.to_csv(index=True)
        st.download_button(
            label="📄 Download VIP Users",
            data=high_value_csv,
            file_name="vip_users.csv",
            mime="text/csv"
        )
    
    with col3:
        product_summary = predictions.groupby('product_id').agg({
            'recommendation_model_prediction': 'mean',
            'reorder_model_prediction': 'mean'
        })
        product_csv = product_summary.to_csv(index=True)
        st.download_button(
            label="📄 Download Product Insights",
            data=product_csv,
            file_name="product_insights.csv",
            mime="text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Dashboard Info:**
- 🔄 Data auto-refreshes on reload
- 📊 All charts are interactive
- 💾 Export options available
- 🎨 Built with Streamlit & Plotly
""")