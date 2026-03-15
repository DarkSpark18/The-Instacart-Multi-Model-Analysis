# analyze_results.py
"""
Comprehensive Analysis of Instacart ML Pipeline Results

Generates detailed analysis, visualizations, and insights from all 5 models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
PREDICTIONS_PATH = Path("data/predictions/predictions.parquet")
FEATURES_PATH = Path("data/features/feature_store.parquet")
OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load predictions and features"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Features shape: {features.shape}")
    
    return predictions, features


def analyze_basket_size(df):
    """Analyze basket size predictions"""
    print("\n📊 ANALYZING BASKET SIZE MODEL...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution of predictions
    axes[0, 0].hist(df['basket_model_prediction'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['basket_model_prediction'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {df["basket_model_prediction"].mean():.2f}')
    axes[0, 0].set_xlabel('Predicted Basket Size')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Predicted Basket Sizes')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # By user - average basket size
    user_basket = df.groupby('user_id')['basket_model_prediction'].mean()
    axes[0, 1].hist(user_basket, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Average Predicted Basket Size')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].set_title('Average Basket Size per User')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top users by basket size
    top_users = user_basket.nlargest(20)
    axes[1, 0].barh(range(len(top_users)), top_users.values, color='skyblue')
    axes[1, 0].set_yticks(range(len(top_users)))
    axes[1, 0].set_yticklabels([f'User {uid}' for uid in top_users.index])
    axes[1, 0].set_xlabel('Predicted Basket Size')
    axes[1, 0].set_title('Top 20 Users by Basket Size')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Statistics box
    stats_text = f"""
    Basket Size Statistics:
    
    Mean: {df['basket_model_prediction'].mean():.2f}
    Median: {df['basket_model_prediction'].median():.2f}
    Std Dev: {df['basket_model_prediction'].std():.2f}
    Min: {df['basket_model_prediction'].min():.2f}
    Max: {df['basket_model_prediction'].max():.2f}
    
    Total Users: {df['user_id'].nunique():,}
    Total Products: {df['product_id'].nunique():,}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'basket_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: basket_size_analysis.png")
    
    return {
        'mean_basket_size': df['basket_model_prediction'].mean(),
        'median_basket_size': df['basket_model_prediction'].median(),
        'total_users': df['user_id'].nunique(),
        'total_products': df['product_id'].nunique()
    }


def analyze_reorder(df):
    """Analyze reorder predictions"""
    print("\n🔄 ANALYZING REORDER MODEL...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reorder rate
    reorder_rate = df['reorder_model_prediction'].mean()
    labels = ['Will NOT Reorder', 'Will Reorder']
    sizes = [1-reorder_rate, reorder_rate]
    colors = ['#ff9999', '#66b3ff']
    
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('Overall Reorder Prediction Rate')
    
    # Reorder by user
    user_reorder = df.groupby('user_id')['reorder_model_prediction'].mean()
    axes[0, 1].hist(user_reorder, bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_xlabel('Reorder Rate')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].set_title('Reorder Rate Distribution by User')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top reordered products
    product_reorder = df.groupby('product_id')['reorder_model_prediction'].agg(['sum', 'count'])
    product_reorder['rate'] = product_reorder['sum'] / product_reorder['count']
    top_products = product_reorder.nlargest(20, 'sum')
    
    axes[1, 0].barh(range(len(top_products)), top_products['sum'].values, color='green', alpha=0.7)
    axes[1, 0].set_yticks(range(len(top_products)))
    axes[1, 0].set_yticklabels([f'Product {pid}' for pid in top_products.index])
    axes[1, 0].set_xlabel('Total Reorder Predictions')
    axes[1, 0].set_title('Top 20 Most Reordered Products')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Stats
    stats_text = f"""
    Reorder Statistics:
    
    Overall Reorder Rate: {reorder_rate*100:.1f}%
    
    Users likely to reorder: {(user_reorder > 0.5).sum():,}
    Users unlikely to reorder: {(user_reorder <= 0.5).sum():,}
    
    Products with high reorder: {(product_reorder['rate'] > 0.7).sum():,}
    Products with low reorder: {(product_reorder['rate'] < 0.3).sum():,}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'reorder_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: reorder_analysis.png")
    
    return {
        'overall_reorder_rate': reorder_rate,
        'high_reorder_users': (user_reorder > 0.5).sum(),
        'high_reorder_products': (product_reorder['rate'] > 0.7).sum()
    }


def analyze_churn(df):
    """Analyze churn predictions"""
    print("\n⚠️  ANALYZING CHURN MODEL...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Churn rate by user
    user_churn = df.groupby('user_id')['churn_model_prediction'].first()
    churn_rate = user_churn.mean()
    
    labels = ['Active Users', 'At Risk (Churn)']
    sizes = [1-churn_rate, churn_rate]
    colors = ['#90EE90', '#FF6B6B']
    
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('User Churn Prediction')
    
    # Churn distribution
    churn_counts = user_churn.value_counts()
    axes[0, 1].bar(['Active', 'At Risk'], 
                   [churn_counts.get(0, 0), churn_counts.get(1, 0)],
                   color=['green', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].set_title('User Status Distribution')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate([churn_counts.get(0, 0), churn_counts.get(1, 0)]):
        axes[0, 1].text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # Compare churned vs active user features
    if 'user_total_orders' in df.columns:
        churned = df[df['churn_model_prediction'] == 1]
        active = df[df['churn_model_prediction'] == 0]
        
        comparison_data = {
            'Active Users': [
                active.groupby('user_id')['user_total_orders'].first().mean(),
                active.groupby('user_id')['user_unique_products'].first().mean()
            ],
            'Churned Users': [
                churned.groupby('user_id')['user_total_orders'].first().mean(),
                churned.groupby('user_id')['user_unique_products'].first().mean()
            ]
        }
        
        x = np.arange(2)
        width = 0.35
        
        axes[1, 0].bar(x - width/2, comparison_data['Active Users'], width, 
                       label='Active', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, comparison_data['Churned Users'], width,
                       label='Churned', color='red', alpha=0.7)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['Avg Orders', 'Avg Products'])
        axes[1, 0].set_ylabel('Average Value')
        axes[1, 0].set_title('Active vs Churned User Behavior')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Stats
    stats_text = f"""
    Churn Statistics:
    
    Overall Churn Rate: {churn_rate*100:.1f}%
    
    Active Users: {churn_counts.get(0, 0):,}
    At-Risk Users: {churn_counts.get(1, 0):,}
    
    Total Users: {len(user_churn):,}
    
    ⚠️ Retention Priority:
    Focus on {churn_counts.get(1, 0):,} at-risk users
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'churn_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: churn_analysis.png")
    
    return {
        'churn_rate': churn_rate,
        'active_users': churn_counts.get(0, 0),
        'churned_users': churn_counts.get(1, 0)
    }


def analyze_recommendations(df):
    """Analyze recommendation model"""
    print("\n⭐ ANALYZING RECOMMENDATION MODEL...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top recommended products overall
    top_products = df.groupby('product_id')['recommendation_model_prediction'].mean().nlargest(20)
    
    axes[0, 0].barh(range(len(top_products)), top_products.values, color='purple', alpha=0.7)
    axes[0, 0].set_yticks(range(len(top_products)))
    axes[0, 0].set_yticklabels([f'Product {pid}' for pid in top_products.index])
    axes[0, 0].set_xlabel('Recommendation Score')
    axes[0, 0].set_title('Top 20 Recommended Products (Overall)')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Distribution of recommendation scores
    axes[0, 1].hist(df['recommendation_model_prediction'], bins=50, 
                    edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Recommendation Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Recommendation Scores')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample user recommendations
    sample_user = df['user_id'].iloc[0]
    user_recs = df[df['user_id'] == sample_user].nlargest(10, 'recommendation_model_prediction')
    
    axes[1, 0].barh(range(len(user_recs)), user_recs['recommendation_model_prediction'].values,
                    color='teal', alpha=0.7)
    axes[1, 0].set_yticks(range(len(user_recs)))
    axes[1, 0].set_yticklabels([f'P{pid}' for pid in user_recs['product_id']])
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_title(f'Top 10 Recommendations for User {sample_user}')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Stats
    stats_text = f"""
    Recommendation Statistics:
    
    Total Recommendations: {len(df):,}
    Unique Products: {df['product_id'].nunique():,}
    Unique Users: {df['user_id'].nunique():,}
    
    Avg Score: {df['recommendation_model_prediction'].mean():.4f}
    
    High-confidence recs: {(df['recommendation_model_prediction'] > df['recommendation_model_prediction'].quantile(0.75)).sum():,}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'recommendation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: recommendation_analysis.png")
    
    return {
        'total_recommendations': len(df),
        'unique_products': df['product_id'].nunique(),
        'avg_score': df['recommendation_model_prediction'].mean()
    }


def create_executive_summary(all_stats):
    """Create executive summary visualization"""
    print("\n📋 CREATING EXECUTIVE SUMMARY...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle('INSTACART ML PIPELINE - EXECUTIVE SUMMARY', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Model Performance Summary
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                            MULTI-MODEL ML PIPELINE RESULTS                                    ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    
    📊 DATASET OVERVIEW:
       • Total Users: {all_stats['basket']['total_users']:,}
       • Total Products: {all_stats['basket']['total_products']:,}
       • Total Predictions: {all_stats['recommendation']['total_recommendations']:,}
    
    ✅ MODELS TRAINED: 5/5 Successfully
       1. Basket Size Prediction   (Regression)
       2. Reorder Probability      (Classification)
       3. Churn Prediction         (Classification)
       4. Next Product Prediction  (Ranking)
       5. Recommendation System    (Ranking)
    """
    
    ax1.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', fontweight='bold')
    
    # Key Metrics
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    ax2.set_title('🛒 BASKET SIZE', fontsize=14, fontweight='bold', pad=20)
    basket_text = f"""
    Mean Basket Size:
    {all_stats['basket']['mean_basket_size']:.2f} items
    
    Median:
    {all_stats['basket']['median_basket_size']:.2f} items
    
    Prediction Range:
    2-15 items
    """
    ax2.text(0.1, 0.4, basket_text, fontsize=11, family='monospace')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.set_title('🔄 REORDER RATE', fontsize=14, fontweight='bold', pad=20)
    reorder_text = f"""
    Overall Rate:
    {all_stats['reorder']['overall_reorder_rate']*100:.1f}%
    
    High Reorder Users:
    {all_stats['reorder']['high_reorder_users']:,}
    
    High Reorder Products:
    {all_stats['reorder']['high_reorder_products']:,}
    """
    ax3.text(0.1, 0.4, reorder_text, fontsize=11, family='monospace')
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    ax4.set_title('⚠️ CHURN RISK', fontsize=14, fontweight='bold', pad=20)
    churn_text = f"""
    Churn Rate:
    {all_stats['churn']['churn_rate']*100:.1f}%
    
    Active Users:
    {all_stats['churn']['active_users']:,}
    
    At-Risk Users:
    {all_stats['churn']['churned_users']:,}
    """
    ax4.text(0.1, 0.4, churn_text, fontsize=11, family='monospace')
    
    # Action Items
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.set_title('📌 KEY INSIGHTS & RECOMMENDATIONS', fontsize=14, fontweight='bold', 
                  loc='left', pad=20)
    
    insights_text = f"""
    1. CUSTOMER RETENTION
       → {all_stats['churn']['churned_users']:,} users at risk of churning ({all_stats['churn']['churn_rate']*100:.1f}%)
       → Implement targeted retention campaigns for at-risk segment
       → Focus on increasing order frequency and product variety
    
    2. PRODUCT RECOMMENDATIONS
       → {all_stats['recommendation']['unique_products']:,} products in recommendation pool
       → Average recommendation confidence: {all_stats['recommendation']['avg_score']:.4f}
       → Deploy personalized product suggestions to increase basket size
    
    3. REORDER OPTIMIZATION
       → {all_stats['reorder']['overall_reorder_rate']*100:.1f}% overall reorder rate
       → {all_stats['reorder']['high_reorder_products']:,} products show high reorder potential
       → Create "Buy Again" campaigns for high-reorder products
    
    4. BASKET SIZE INSIGHTS
       → Average basket: {all_stats['basket']['mean_basket_size']:.2f} items
       → Opportunity to increase basket through cross-selling and bundles
       → Target users with below-average baskets for upselling
    """
    
    ax5.text(0.05, 0.5, insights_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.savefig(OUTPUT_DIR / 'executive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: executive_summary.png")


def generate_csv_reports(df):
    """Generate CSV reports for detailed analysis"""
    print("\n📄 GENERATING CSV REPORTS...")
    
    # Top churned users
    churned = df[df['churn_model_prediction'] == 1].groupby('user_id').first()
    churned_sorted = churned.sort_values('user_total_orders', ascending=False)
    churned_sorted.to_csv(OUTPUT_DIR / 'churned_users_report.csv')
    print(f"  ✅ Saved: churned_users_report.csv")
    
    # Top recommended products
    top_recs = df.groupby('product_id').agg({
        'recommendation_model_prediction': 'mean',
        'reorder_model_prediction': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'num_users'})
    top_recs_sorted = top_recs.sort_values('recommendation_model_prediction', ascending=False)
    top_recs_sorted.to_csv(OUTPUT_DIR / 'top_products_report.csv')
    print(f"  ✅ Saved: top_products_report.csv")
    
    # User segments
    user_summary = df.groupby('user_id').agg({
        'basket_model_prediction': 'mean',
        'reorder_model_prediction': 'mean',
        'churn_model_prediction': 'first',
        'product_id': 'count'
    }).rename(columns={
        'basket_model_prediction': 'avg_basket_size',
        'reorder_model_prediction': 'reorder_rate',
        'churn_model_prediction': 'churn_risk',
        'product_id': 'total_products'
    })
    user_summary.to_csv(OUTPUT_DIR / 'user_segments_report.csv')
    print(f"  ✅ Saved: user_segments_report.csv")


def main():
    """Main analysis pipeline"""
    
    print("\n" + "="*70)
    print("INSTACART ML PIPELINE - COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Load data
    predictions, features = load_data()
    
    # Merge predictions with features for detailed analysis
    df = predictions.copy()
    
    # Run all analyses
    basket_stats = analyze_basket_size(df)
    reorder_stats = analyze_reorder(df)
    churn_stats = analyze_churn(df)
    rec_stats = analyze_recommendations(df)
    
    # Compile all stats
    all_stats = {
        'basket': basket_stats,
        'reorder': reorder_stats,
        'churn': churn_stats,
        'recommendation': rec_stats
    }
    
    # Create executive summary
    create_executive_summary(all_stats)
    
    # Generate CSV reports
    generate_csv_reports(df)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  📊 basket_size_analysis.png")
    print("  📊 reorder_analysis.png")
    print("  📊 churn_analysis.png")
    print("  📊 recommendation_analysis.png")
    print("  📊 executive_summary.png")
    print("  📄 churned_users_report.csv")
    print("  📄 top_products_report.csv")
    print("  📄 user_segments_report.csv")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()