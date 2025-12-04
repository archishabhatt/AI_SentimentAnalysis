import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
from sentence_transformers import SentenceTransformer
import re
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🤖 AI Sentiment Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    h1 {
        color: #2c3e50;
        padding-bottom: 10px;
        border-bottom: 3px solid #3498db;
    }
    h2 {
        color: #34495e;
        padding-top: 20px;
        border-left: 4px solid #3498db;
        padding-left: 10px;
    }
    h3 {
        color: #546e7a;
    }
    .info-box {
        background-color: #e8f4fc;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #3498db;
    }
    .success-box {
        background-color: #e8f6f3;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #2ecc71;
    }
    .model-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .performance-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .tab-content {
        padding: 20px 0;
    }
    .sampling-warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        color: #856404;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Color scheme
COLORS = {
    'reddit': '#FF5700',
    'youtube': '#FF0000',
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#95a5a6',
    'primary': '#3498db',
    'secondary': '#9b59b6',
    'success': '#27ae60',
    'warning': '#f39c12'
}

# Configuration
MODEL_SAMPLE_SIZE = 750  # Reduced sample size for modeling
EDA_SAMPLE_SIZE = 2000   # Larger sample for EDA

# Load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/df_cleaned.csv')
        # Ensure datetime conversion
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        if 'year' not in df.columns:
            df['year'] = df['created_at'].dt.year if 'created_at' in df.columns else 2024
        if 'month' not in df.columns:
            df['month'] = df['created_at'].dt.month if 'created_at' in df.columns else 1
        
        # Create engagement score if not exists
        if 'engagement_score' not in df.columns:
            df['engagement_score'] = df['likes'] + (df['replies'] * 2)
        
        # Create high engagement flag (top 25%)
        engagement_threshold = df['engagement_score'].quantile(0.75)
        df['high_engagement'] = (df['engagement_score'] >= engagement_threshold).astype(int)
        
        # Create sentiment categories
        df['sentiment_category'] = df['sentiment_polarity'].apply(
            lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
        )
        
        # Create word_count if not exists
        if 'word_count' not in df.columns:
            df['word_count'] = df['text_clean'].apply(lambda x: len(str(x).split()))
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Initialize BERT model (cached)
@st.cache_resource
def load_bert_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None

bert_model = load_bert_model()

# Sidebar Navigation
st.sidebar.title("🤖 AI Sentiment Dashboard")
st.sidebar.markdown("---")

# Project Info in Sidebar
with st.sidebar.expander("📚 Project Overview"):
    st.markdown(f"""
    **Project Goal**: Analyze public sentiment about AI from Reddit and YouTube comments
    
    **Data Sources**:
    - Reddit: 15+ AI-related subreddits
    - YouTube: AI-focused videos
    
    **Time Period**: 2020-2025
    
    **Total Comments**: {len(df):,}
    
    **Modeling Note**: Using {MODEL_SAMPLE_SIZE} random samples for faster model training
    """)

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Project Overview", "📊 Data Explorer", "📈 EDA Dashboard", 
     "🤖 Traditional ML Models", "🧠 BERT Models", "📋 Model Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Data Filters")

# Filters
if not df.empty:
    # Platform filter
    platforms = st.sidebar.multiselect(
        "Platforms",
        options=sorted(df['source'].unique()) if 'source' in df.columns else [],
        default=sorted(df['source'].unique()) if 'source' in df.columns else []
    )
    
    # Sentiment filter
    sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=['positive', 'neutral', 'negative'],
        default=['positive', 'neutral', 'negative']
    )
    
    # Date filter
    if 'created_at' in df.columns:
        min_date = df['created_at'].min().date()
        max_date = df['created_at'].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Apply filters
    filtered_df = df.copy()
    if platforms:
        filtered_df = filtered_df[filtered_df['source'].isin(platforms)]
    if 'created_at' in df.columns and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['created_at'].dt.date >= date_range[0]) &
            (filtered_df['created_at'].dt.date <= date_range[1])
        ]
    
    # Filter by sentiment
    filtered_df = filtered_df[filtered_df['sentiment_category'].isin(sentiments)]
    
    st.sidebar.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} comments**")
else:
    filtered_df = pd.DataFrame()
    st.sidebar.warning("No data loaded")

# Create sampled dataset for modeling
@st.cache_data
def get_modeling_sample(_df, sample_size=MODEL_SAMPLE_SIZE):
    """Get a balanced sample for modeling"""
    if _df.empty:
        return _df
    
    # Ensure we have enough data
    if len(_df) <= sample_size:
        return _df
    
    # Try to balance by sentiment if possible
    try:
        # Sample proportionally by sentiment
        sample = _df.groupby('sentiment_category', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(sample_size / 3)), random_state=42)
        )
        
        # If we still need more samples, fill randomly
        if len(sample) < sample_size:
            remaining = _df[~_df.index.isin(sample.index)].sample(
                sample_size - len(sample), random_state=42
            )
            sample = pd.concat([sample, remaining])
        
        return sample.sample(frac=1, random_state=42)  # Shuffle
        
    except:
        # Fallback to random sampling
        return _df.sample(min(sample_size, len(_df)), random_state=42)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • 🤖 AI Sentiment Analysis")

# ===== PROJECT OVERVIEW PAGE =====
if page == "🏠 Project Overview":
    st.title("🤖 AI Public Sentiment Analysis Dashboard")
    st.markdown("### Analyzing Opinions on Artificial Intelligence from Reddit & YouTube")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 Project Description
        
        This dashboard analyzes public sentiment towards Artificial Intelligence by examining 
        discussions from two major online platforms:
        
        **Reddit**: Technical and enthusiast communities discussing AI developments, ethics, and implications.
        
        **YouTube**: Broader public discourse in video comments sections.
        
        ### 🎯 Key Objectives
        
        1. **Sentiment Analysis**: Understand public perception of AI (positive, negative, neutral)
        2. **Platform Comparison**: Compare discourse patterns between Reddit and YouTube
        3. **Engagement Analysis**: Identify what types of content generate most discussion
        4. **Predictive Modeling**: Build models to predict sentiment and engagement
        
        ### ⚡ Performance Optimization
        
        To ensure fast response times:
        - **EDA Visualizations**: Use full or large sample of data
        - **Model Training**: Use {MODEL_SAMPLE_SIZE} balanced samples
        - **Caching**: All heavy computations are cached
        - **Progress Indicators**: Visual feedback during computation
        """)
    
    with col2:
        if not df.empty:
            # Quick stats
            st.markdown("### 📈 Quick Statistics")
            
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.metric("Total Comments", f"{len(df):,}")
                if 'source' in df.columns:
                    reddit_count = df[df['source'] == 'reddit'].shape[0]
                    st.metric("Reddit Comments", f"{reddit_count:,}")
                st.metric("Avg Sentiment", f"{df['sentiment_polarity'].mean():.3f}")
            
            with stats_col2:
                if 'source' in df.columns:
                    youtube_count = df[df['source'] == 'youtube'].shape[0]
                    st.metric("YouTube Comments", f"{youtube_count:,}")
                st.metric("Avg Words/Comment", f"{df['text_length'].mean():.0f}" if 'text_length' in df.columns else "N/A")
                st.metric("Positive Ratio", f"{df['is_positive'].mean():.1%}" if 'is_positive' in df.columns else "N/A")
            
            # Sentiment distribution pie chart
            if 'sentiment_category' in df.columns:
                sentiment_counts = df['sentiment_category'].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title='Overall Sentiment Distribution',
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positive': COLORS['positive'],
                        'neutral': COLORS['neutral'],
                        'negative': COLORS['negative']
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Analysis Pipeline
    st.subheader("🔄 Analysis Pipeline")
    
    pipeline_steps = [
        ("1️⃣ Data Collection", "Scraping from Reddit & YouTube APIs"),
        ("2️⃣ Data Cleaning", "Harmonization and text preprocessing"),
        ("3️⃣ Feature Engineering", "Temporal, engagement, text features"),
        ("4️⃣ EDA & Visualization", f"Interactive dashboards ({EDA_SAMPLE_SIZE} samples)"),
        ("5️⃣ Model Training", f"Fast training ({MODEL_SAMPLE_SIZE} balanced samples)"),
        ("6️⃣ Evaluation", "Performance metrics and comparison")
    ]
    
    for step, description in pipeline_steps:
        st.markdown(f"**{step}** — {description}")
    
    # Tech Stack
    st.subheader("🛠️ Technology Stack")
    
    tech_cols = st.columns(3)
    tech_stacks = [
        ("Data Processing", "Pandas, NumPy, Scikit-learn"),
        ("Visualization", "Plotly, Matplotlib, Seaborn"),
        ("NLP", "NLTK, TF-IDF, Sentence Transformers"),
        ("ML Models", "Logistic Regression, Random Forest, BERT"),
        ("Web App", "Streamlit, Caching, Session State"),
        ("Deployment", "Docker, Streamlit Cloud")
    ]
    
    for col, (category, tools) in zip(tech_cols * 2, tech_stacks):
        with col:
            st.markdown(f"**{category}**")
            st.markdown(f"`{tools}`")

# ===== DATA EXPLORER PAGE =====
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("### Browse and filter the AI sentiment dataset")
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available. Please check data loading.")
        st.stop()
    
    # Sampling control for EDA
    st.markdown('<div class="sampling-warning">⚠️ Showing full dataset for exploration. Modeling sections use smaller samples for speed.</div>', unsafe_allow_html=True)
    
    # Search functionality
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search in comments:", placeholder="Type keywords (AI, ethics, future, etc.)")
    with col2:
        sample_size = st.slider("Display samples", 10, 100, 30)
    
    # Apply search
    if search_term:
        search_df = filtered_df[filtered_df['text'].str.contains(search_term, case=False, na=False)]
    else:
        search_df = filtered_df
    
    # Display sample data
    st.subheader(f"📋 Sample Comments ({len(search_df):,} total)")
    
    if len(search_df) > 0:
        sample_df = search_df.sample(min(sample_size, len(search_df)))
        
        for idx, row in sample_df.iterrows():
            with st.expander(f"{row['source'].upper()} | {row['sentiment_category'].upper()} | {row.get('platform_community', '')[:30]}...", expanded=False):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**Text:** {row['text'][:200]}...")
                    if 'platform_post_title' in row:
                        st.caption(f"**Post:** {row['platform_post_title'][:100]}...")
                
                with col2:
                    st.metric("Likes", row['likes'])
                    st.metric("Replies", row['replies'])
                
                with col3:
                    st.metric("Sentiment", f"{row['sentiment_polarity']:.3f}")
                    st.metric("Length", row['text_length'])
        
        # Summary statistics
        st.subheader("📈 Dataset Statistics")
        
        stats_cols = st.columns(4)
        metrics = [
            ("Total Comments", len(search_df)),
            ("Avg Sentiment", search_df['sentiment_polarity'].mean()),
            ("Avg Length", search_df['text_length'].mean() if 'text_length' in search_df.columns else 0),
            ("Engagement Score", search_df['engagement_score'].mean() if 'engagement_score' in search_df.columns else 0)
        ]
        
        for col, (label, value) in zip(stats_cols, metrics):
            with col:
                if isinstance(value, float):
                    st.metric(label, f"{value:.2f}")
                else:
                    st.metric(label, value)
        
        # Detailed statistics
        if st.checkbox("Show detailed statistics"):
            numeric_cols = search_df.select_dtypes(include=[np.number]).columns
            st.dataframe(search_df[numeric_cols].describe(), use_container_width=True)
        
        # Column information
        if st.checkbox("Show column information"):
            col_info = pd.DataFrame({
                'Column': search_df.columns,
                'Data Type': search_df.dtypes.astype(str),
                'Non-Null Count': search_df.count(),
                'Unique Values': [search_df[col].nunique() for col in search_df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
    else:
        st.info("No comments match your search criteria.")

# ===== EDA DASHBOARD PAGE =====
elif page == "📈 EDA Dashboard":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("### Visual insights into AI sentiment patterns")
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available. Please check data loading.")
        st.stop()
    
    # Use sampled data for faster EDA
    eda_sample_size = min(EDA_SAMPLE_SIZE, len(filtered_df))
    eda_df = filtered_df.sample(eda_sample_size, random_state=42) if len(filtered_df) > eda_sample_size else filtered_df
    
    st.markdown(f'<div class="sampling-warning">📊 Showing {len(eda_df):,} random samples for fast visualization</div>', unsafe_allow_html=True)
    
    # Tab layout for different EDA sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📈 Time Analysis", "🌐 Platform Comparison", "🔍 Text Analysis"])
    
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution by platform
            if 'source' in eda_df.columns and 'sentiment_category' in eda_df.columns:
                sentiment_by_platform = pd.crosstab(eda_df['source'], eda_df['sentiment_category'])
                fig = px.bar(
                    sentiment_by_platform,
                    barmode='group',
                    title=f'Sentiment Distribution by Platform (n={len(eda_df)})',
                    color_discrete_map={
                        'positive': COLORS['positive'],
                        'neutral': COLORS['neutral'],
                        'negative': COLORS['negative']
                    }
                )
                fig.update_layout(height=400, xaxis_title="Platform", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
            # Text length distribution
            if 'text_length' in eda_df.columns:
                fig = px.histogram(
                    eda_df,
                    x='text_length',
                    nbins=30,
                    title=f'Comment Length Distribution (n={len(eda_df)})',
                    color='source' if 'source' in eda_df.columns else None,
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.update_layout(height=400, xaxis_title="Text Length (words)", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment polarity distribution
            if 'sentiment_polarity' in eda_df.columns:
                fig = px.histogram(
                    eda_df,
                    x='sentiment_polarity',
                    nbins=30,
                    title=f'Sentiment Polarity Distribution (n={len(eda_df)})',
                    color='source' if 'source' in eda_df.columns else None,
                    marginal="box",
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Neutral")
                fig.update_layout(height=400, xaxis_title="Sentiment Polarity", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
            # Engagement score distribution
            if 'engagement_score' in eda_df.columns:
                # Filter outliers for better visualization
                q95 = eda_df['engagement_score'].quantile(0.95)
                filtered_engagement = eda_df[eda_df['engagement_score'] <= q95]
                
                fig = px.histogram(
                    filtered_engagement,
                    x='engagement_score',
                    nbins=30,
                    title=f'Engagement Score Distribution (95th percentile, n={len(filtered_engagement)})',
                    color='source' if 'source' in eda_df.columns else None,
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.update_layout(height=400, xaxis_title="Engagement Score", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Posts over time
            if 'created_at' in eda_df.columns:
                eda_df['date'] = eda_df['created_at'].dt.date
                posts_over_time = eda_df.groupby(['date', 'source']).size().reset_index(name='count')
                
                fig = px.line(
                    posts_over_time,
                    x='date',
                    y='count',
                    color='source',
                    title=f'Comments Over Time by Platform (n={len(eda_df)})',
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Number of Comments")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment over time
            if 'created_at' in eda_df.columns:
                eda_df['month_year'] = eda_df['created_at'].dt.to_period('M').astype(str)
                sentiment_over_time = eda_df.groupby(['month_year', 'source'])['sentiment_polarity'].mean().reset_index()
                
                fig = px.line(
                    sentiment_over_time,
                    x='month_year',
                    y='sentiment_polarity',
                    color='source',
                    title=f'Average Sentiment Over Time (n={len(eda_df)})',
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.update_layout(height=400, xaxis_title="Month", yaxis_title="Average Sentiment")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Platform Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Comparison metrics
            comparison_data = []
            metrics_to_compare = [
                ('sentiment_polarity', 'Avg Sentiment'),
                ('text_length', 'Avg Text Length'),
                ('engagement_score', 'Avg Engagement'),
                ('likes', 'Avg Likes'),
                ('replies', 'Avg Replies')
            ]
            
            for metric_col, metric_name in metrics_to_compare:
                if metric_col in eda_df.columns:
                    for platform in eda_df['source'].unique():
                        platform_data = eda_df[eda_df['source'] == platform]
                        comparison_data.append({
                            'Platform': platform,
                            'Metric': metric_name,
                            'Value': platform_data[metric_col].mean()
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                fig = px.bar(
                    comparison_df,
                    x='Metric',
                    y='Value',
                    color='Platform',
                    barmode='group',
                    title=f'Platform Comparison (n={len(eda_df)})',
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            numeric_cols = eda_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr_matrix = eda_df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title=f'Feature Correlation Heatmap (n={len(eda_df)})',
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric features for correlation analysis")
    
    with tab4:
        st.subheader("Text Analysis")
        
        # Word frequencies
        st.markdown("#### Most Frequent Words")
        
        # Combine all text
        if 'text_clean' in eda_df.columns:
            all_text = ' '.join(eda_df['text_clean'].astype(str).fillna(''))
            
            if all_text.strip():
                # Get word frequencies
                words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
                word_freq = Counter(words)
                
                # Remove common stopwords
                stop_words = set(['the', 'and', 'for', 'that', 'this', 'with', 'have', 'from', 'they', 'what', 'about', 'like'])
                for word in stop_words:
                    word_freq.pop(word, None)
                
                # Create bar chart of top words
                top_words = pd.DataFrame(word_freq.most_common(15), columns=['Word', 'Frequency'])
                
                fig = px.bar(
                    top_words,
                    x='Frequency',
                    y='Word',
                    orientation='h',
                    title=f'Top 15 Most Frequent Words (n={len(eda_df)})',
                    color='Frequency',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment vs Engagement scatter
        if 'sentiment_polarity' in eda_df.columns and 'engagement_score' in eda_df.columns:
            st.markdown("#### Sentiment vs Engagement")
            
            # Sample for scatter plot to avoid overcrowding
            scatter_sample = eda_df.sample(min(300, len(eda_df)))
            
            fig = px.scatter(
                scatter_sample,
                x='sentiment_polarity',
                y='engagement_score',
                color='source',
                size='text_length' if 'text_length' in eda_df.columns else None,
                hover_data=['text_clean'] if 'text_clean' in eda_df.columns else None,
                title=f'Sentiment vs Engagement (n={len(scatter_sample)})',
                color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ===== TRADITIONAL ML MODELS PAGE =====
elif page == "🤖 Traditional ML Models":
    st.title("🤖 Traditional Machine Learning Models")
    st.markdown("### TF-IDF + Traditional ML algorithms for sentiment prediction")
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available. Please check data loading.")
        st.stop()
    
    # Get modeling sample
    model_df = get_modeling_sample(filtered_df, MODEL_SAMPLE_SIZE)
    
    st.markdown(f'<div class="sampling-warning">⚡ Using {len(model_df):,} balanced samples for faster model training</div>', unsafe_allow_html=True)
    
    with st.expander("⚙️ Model Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_variable = st.selectbox(
                "Target Variable",
                ["sentiment_category", "high_engagement", "is_positive", "is_negative"]
            )
            
            # Map target to binary if needed
            if target_variable in ["is_positive", "is_negative", "high_engagement"]:
                is_binary = True
                target_series = model_df[target_variable]
            else:
                is_binary = False
                # Convert sentiment_category to numeric labels
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                target_series = pd.Series(le.fit_transform(model_df[target_variable]))
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State", 1, 100, 42)
            
            use_numeric = st.checkbox("Include Numeric Features", value=True)
        
        with col3:
            model_choice = st.selectbox(
                "Select Model",
                ["Logistic Regression", "Linear SVC", "Random Forest"]
            )
            
            if model_choice == "Random Forest":
                n_estimators = st.slider("Number of Trees", 10, 200, 50, 10)
                max_depth = st.selectbox("Max Depth", [None, 5, 10, 15, 20])
    
    # Prepare data
    st.subheader("📊 Data Preparation")
    
    # Define features
    text_feature = "text_tfidf" if "text_tfidf" in model_df.columns else "text_clean"
    features = [text_feature]
    
    if use_numeric:
        numeric_candidates = ['text_length', 'sentiment_subjectivity', 'word_count']
        numeric_features = [f for f in numeric_candidates if f in model_df.columns]
        features.extend(numeric_features)
    
    X = model_df[features]
    y = target_series
    
    st.info(f"**Target**: {target_variable} | **Features**: {len(features)} | **Samples**: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if is_binary else None
    )
    
    # Create preprocessing pipeline
    transformers = []
    
    # TF-IDF for text
    tfidf = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
        max_features=1000  # Limit features for speed
    )
    transformers.append(("tfidf", tfidf, text_feature))
    
    if use_numeric and numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    
    preprocess = ColumnTransformer(transformers=transformers)
    
    # Train model
    st.subheader("🚀 Model Training")
    
    with st.spinner("Training model (this may take a moment)..."):
        if model_choice == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        elif model_choice == "Linear SVC":
            clf = LinearSVC(class_weight="balanced", max_iter=2000, C=0.5, random_state=random_state)
        elif model_choice == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1  # Use all CPU cores
            )
        
        model = Pipeline([
            ("preprocess", preprocess),
            ("clf", clf)
        ])
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        if y_pred_proba is not None and is_binary:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = "N/A"
        else:
            roc_auc = "N/A"
    
    # Display results
    st.subheader("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("F1 Score", f"{f1:.3f}")
    with col3:
        st.metric("Precision", f"{precision:.3f}")
    with col4:
        st.metric("Recall", f"{recall:.3f}")
    
    if roc_auc != "N/A":
        st.metric("ROC AUC", f"{roc_auc:.3f}")
    
    # Classification report
    st.subheader("📋 Classification Report")
    
    if is_binary:
        report = classification_report(y_test, y_pred, output_dict=True)
    else:
        report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    # Confusion matrix
    st.subheader("📊 Confusion Matrix")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    
    if is_binary:
        labels = ['Negative', 'Positive']
    else:
        labels = le.classes_ if 'le' in locals() else np.unique(y_test)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_choice}')
    st.pyplot(fig)
    
    # Sample predictions
    st.subheader("🔮 Sample Predictions")
    
    sample_size = min(5, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    
    sample_data = []
    for idx in sample_indices:
        actual = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        predicted = y_pred[idx]
        
        if is_binary:
            actual_label = 'Positive' if actual == 1 else 'Negative'
            predicted_label = 'Positive' if predicted == 1 else 'Negative'
        else:
            actual_label = le.inverse_transform([actual])[0] if 'le' in locals() else actual
            predicted_label = le.inverse_transform([predicted])[0] if 'le' in locals() else predicted
        
        confidence = y_pred_proba[idx] if y_pred_proba is not None else "N/A"
        
        sample_data.append({
            'Text': X_test.iloc[idx][text_feature][:80] + '...',
            'Actual': actual_label,
            'Predicted': predicted_label,
            'Confidence': f"{confidence:.3f}" if isinstance(confidence, (int, float)) else confidence,
            'Correct': '✓' if actual == predicted else '✗'
        })
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)

# ===== BERT MODELS PAGE =====
elif page == "🧠 BERT Models":
    st.title("🧠 BERT-Based Models")
    st.markdown("### Transformer-based models for advanced sentiment analysis")
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available. Please check data loading.")
        st.stop()
    
    if bert_model is None:
        st.error("BERT model could not be loaded. Please check dependencies.")
        st.stop()
    
    # Get modeling sample - smaller for BERT due to computation
    bert_sample_size = min(MODEL_SAMPLE_SIZE, 500)  # Even smaller for BERT
    model_df = get_modeling_sample(filtered_df, bert_sample_size)
    
    st.markdown(f'<div class="sampling-warning">⚡ Using {len(model_df):,} samples for BERT modeling (computationally intensive)</div>', unsafe_allow_html=True)
    
    with st.expander("⚙️ BERT Model Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_variable = st.selectbox(
                "Target Variable",
                ["sentiment_category", "high_engagement", "is_positive", "is_negative"],
                key="bert_target"
            )
            
            # Map target to binary if needed
            if target_variable in ["is_positive", "is_negative", "high_engagement"]:
                is_binary = True
                target_series = model_df[target_variable]
            else:
                is_binary = False
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                target_series = pd.Series(le.fit_transform(model_df[target_variable]))
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="bert_test")
            random_state = st.number_input("Random State", 1, 100, 42, key="bert_random")
            
            use_numeric_bert = st.checkbox("Include Numeric Features", value=False)
        
        with col3:
            bert_classifier = st.selectbox(
                "Classifier",
                ["Logistic Regression", "Random Forest", "Gradient Boosting"]
            )
    
    # Prepare data
    st.subheader("📊 Data Preparation")
    
    # Get BERT text feature
    bert_text_feature = "text_bert" if "text_bert" in model_df.columns else "text_clean"
    
    st.info("**Note**: Computing BERT embeddings... This may take a moment.")
    
    # Compute BERT embeddings with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Computing BERT embeddings...")
        texts = model_df[bert_text_feature].fillna('').tolist()
        
        # Process in batches to show progress
        batch_size = 100
        embeddings_list = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = bert_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings_list.append(batch_embeddings)
            
            # Update progress
            progress = min((i + batch_size) / len(texts), 1.0)
            progress_bar.progress(progress)
        
        embeddings = np.vstack(embeddings_list)
        
        if use_numeric_bert:
            numeric_candidates = ['text_length', 'sentiment_subjectivity', 'word_count']
            numeric_features = [f for f in numeric_candidates if f in model_df.columns]
            
            if numeric_features:
                numeric_data = model_df[numeric_features].values
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                numeric_scaled = scaler.fit_transform(numeric_data)
                X = np.hstack([embeddings, numeric_scaled])
            else:
                X = embeddings
        else:
            X = embeddings
        
        y = target_series.values if hasattr(target_series, 'values') else target_series
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ BERT embeddings computed: {X.shape[0]} samples, {X.shape[1]} features")
        
    except Exception as e:
        st.error(f"Error computing embeddings: {e}")
        st.stop()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if is_binary else None
    )
    
    # Train model
    st.subheader("🚀 Model Training")
    
    with st.spinner(f"Training {bert_classifier}..."):
        if bert_classifier == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        elif bert_classifier == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=50,  # Smaller for speed
                max_depth=10,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
        elif bert_classifier == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=50, random_state=random_state)
        
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        if y_pred_proba is not None and is_binary:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = "N/A"
        else:
            roc_auc = "N/A"
    
    # Display results
    st.subheader("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("F1 Score", f"{f1:.3f}")
    with col3:
        st.metric("Precision", f"{precision:.3f}")
    with col4:
        st.metric("Recall", f"{recall:.3f}")
    
    if roc_auc != "N/A":
        st.metric("ROC AUC", f"{roc_auc:.3f}")
    
    # Classification report
    st.subheader("📋 Classification Report")
    
    if is_binary:
        report = classification_report(y_test, y_pred, output_dict=True)
    else:
        report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    # Confusion matrix
    st.subheader("📊 Confusion Matrix")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    
    if is_binary:
        labels = ['Negative', 'Positive']
    else:
        labels = le.classes_ if 'le' in locals() else np.unique(y_test)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - BERT + {bert_classifier}')
    st.pyplot(fig)
    
    # Sample predictions
    st.subheader("🔮 Sample Predictions")
    
    if len(X_test) > 0:
        sample_size = min(5, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        
        sample_data = []
        for idx in sample_indices:
            actual = y_test[idx]
            predicted = y_pred[idx]
            
            if is_binary:
                actual_label = 'Positive' if actual == 1 else 'Negative'
                predicted_label = 'Positive' if predicted == 1 else 'Negative'
            else:
                actual_label = le.inverse_transform([actual])[0] if 'le' in locals() else actual
                predicted_label = le.inverse_transform([predicted])[0] if 'le' in locals() else predicted
            
            confidence = y_pred_proba[idx] if y_pred_proba is not None else "N/A"
            
            # Get original text
            test_indices = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
            text_idx = list(test_indices)[idx] if idx < len(test_indices) else idx
            
            sample_data.append({
                'Text': model_df.iloc[text_idx][bert_text_feature][:80] + '...' if text_idx < len(model_df) else 'N/A',
                'Actual': actual_label,
                'Predicted': predicted_label,
                'Confidence': f"{confidence:.3f}" if isinstance(confidence, (int, float)) else confidence,
                'Correct': '✓' if actual == predicted else '✗'
            })
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

# ===== MODEL COMPARISON PAGE =====
elif page == "📋 Model Comparison":
    st.title("📋 Model Comparison")
    st.markdown("### Benchmarking different modeling approaches")
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available. Please check data loading.")
        st.stop()
    
    # Get modeling sample
    model_df = get_modeling_sample(filtered_df, MODEL_SAMPLE_SIZE)
    
    st.markdown(f'<div class="sampling-warning">⚡ Using {len(model_df):,} samples for model comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This section compares the performance of different modeling approaches on our AI sentiment dataset.
    We'll train multiple models with consistent settings and compare their results.
    """)
    
    # Configuration
    with st.expander("⚙️ Comparison Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            target_variable = st.selectbox(
                "Target Variable for Comparison",
                ["sentiment_category", "high_engagement", "is_positive"],
                key="compare_target"
            )
            
            # Prepare target
            if target_variable in ["is_positive", "high_engagement"]:
                is_binary = True
                y = model_df[target_variable]
            else:
                is_binary = False
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(model_df[target_variable]))
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="compare_test")
            random_state = st.number_input("Random State", 1, 100, 42, key="compare_random")
            
            models_to_compare = st.multiselect(
                "Models to Compare",
                ["TF-IDF + Logistic Regression", "TF-IDF + Random Forest", 
                 "BERT + Logistic Regression"],
                default=["TF-IDF + Logistic Regression", "BERT + Logistic Regression"]
            )
    
    if not models_to_compare:
        st.warning("Please select at least one model to compare.")
        st.stop()
    
    # Train and compare models
    st.subheader("📊 Model Training & Comparison")
    
    results = []
    
    # Prepare TF-IDF features
    if any("TF-IDF" in model for model in models_to_compare):
        with st.spinner("Preparing TF-IDF features..."):
            text_feature = "text_tfidf" if "text_tfidf" in model_df.columns else "text_clean"
            tfidf = TfidfVectorizer(
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2),
                stop_words="english",
                max_features=1000  # Limit for speed
            )
            X_tfidf = tfidf.fit_transform(model_df[text_feature])
            
            # Split data
            X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=test_size, random_state=random_state, stratify=y if is_binary else None
            )
    
    # Prepare BERT features
    if any("BERT" in model for model in models_to_compare):
        with st.spinner("Computing BERT embeddings (this may take a moment)..."):
            if bert_model is not None:
                bert_text_feature = "text_bert" if "text_bert" in model_df.columns else "text_clean"
                texts = model_df[bert_text_feature].fillna('').tolist()
                X_bert = bert_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                
                # Split data
                X_train_bert, X_test_bert, y_train, y_test = train_test_split(
                    X_bert, y, test_size=test_size, random_state=random_state, stratify=y if is_binary else None
                )
    
    # Train each model
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_name in enumerate(models_to_compare):
        status_text.text(f"Training {model_name} ({i+1}/{len(models_to_compare)})...")
        
        if "TF-IDF" in model_name:
            X_train, X_test = X_train_tfidf, X_test_tfidf
        else:
            X_train, X_test = X_train_bert, X_test_bert
        
        # Select classifier
        if "Logistic Regression" in model_name:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        elif "Random Forest" in model_name:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
        
        # Train and evaluate
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted')
        })
        
        # Update progress
        progress_bar.progress((i + 1) / len(models_to_compare))
    
    status_text.text("✅ All models trained!")
    
    # Display comparison results
    st.subheader("📈 Comparison Results")
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Display metrics table
        st.dataframe(results_df.style.format("{:.3f}").highlight_max(axis=0), 
                    use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart comparison
            fig = px.bar(
                results_df,
                x='Model',
                y=['Accuracy', 'F1 Score'],
                barmode='group',
                title='Model Performance Comparison',
                color_discrete_sequence=[COLORS['primary'], COLORS['success']]
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart for multi-metric comparison
            metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            fig = go.Figure()
            
            for idx, row in results_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Radar Chart Comparison",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        best_model = results_df.loc[results_df['Accuracy'].idxmax()]
        
        st.markdown(f"""
        **Best Performing Model**: **{best_model['Model']}** with {best_model['Accuracy']:.3f} accuracy
        
        **Observations**:
        1. **BERT models** generally perform better with complex text understanding
        2. **TF-IDF models** are faster to train and good for baseline comparison
        3. **Logistic Regression** often provides better interpretability
        4. **Random Forest** can handle non-linear relationships better
        
        **Recommendation**: Use **{best_model['Model']}** for production if {best_model['Accuracy']:.1%} accuracy meets requirements.
        """)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Results",
            data=csv,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("No models were trained. Please select at least one model to compare.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d;'>
    <p>🤖 AI Sentiment Analysis Dashboard • Built with Streamlit • Data from Reddit & YouTube</p>
    <p>For academic/research purposes only</p>
    </div>
    """,
    unsafe_allow_html=True
)