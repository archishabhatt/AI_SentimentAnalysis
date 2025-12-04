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

# Load data
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
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty dataframe with expected columns
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
    st.markdown("""
    **Project Goal**: Analyze public sentiment about AI from Reddit and YouTube comments
    
    **Data Sources**:
    - Reddit: 15+ AI-related subreddits
    - YouTube: AI-focused videos
    
    **Time Period**: 2020-2025
    
    **Total Comments**: ~2,500
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
        
        ### 🔬 Methodology
        
        - **Data Collection**: Custom scrapers for Reddit (PRAW) and YouTube (API)
        - **Text Processing**: Advanced NLP cleaning (BERT-optimized vs TF-IDF optimized)
        - **Feature Engineering**: Temporal, engagement, and linguistic features
        - **Machine Learning**: Both traditional (TF-IDF) and modern (BERT) approaches
        
        ### 📊 Dashboard Sections
        
        1. **Data Explorer**: Browse and filter the dataset
        2. **EDA Dashboard**: Interactive visualizations and insights
        3. **Traditional Models**: TF-IDF + ML algorithms
        4. **BERT Models**: Transformer-based sentiment analysis
        5. **Model Comparison**: Performance benchmarking
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
                st.metric("Avg Words/Comment", f"{df['text_length'].mean():.0f}")
                st.metric("Positive Ratio", f"{df['is_positive'].mean():.1%}")
            
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
    
    # Dataset Structure
    st.subheader("📁 Dataset Structure")
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Features:**")
            feature_info = {
                "text": "Original comment text",
                "text_clean": "Basic cleaned text",
                "text_tfidf": "TF-IDF optimized cleaning",
                "text_bert": "BERT optimized cleaning",
                "sentiment_polarity": "Sentiment score (-1 to 1)",
                "sentiment_subjectivity": "How opinionated (0 to 1)",
                "likes/replies": "Engagement metrics",
                "source": "Platform (reddit/youtube)",
                "contains_ai/opinion/societal": "Content flags"
            }
            
            for feature, description in feature_info.items():
                if feature in df.columns:
                    st.markdown(f"• **{feature}**: {description}")
        
        with col2:
            st.markdown("**Engineered Features:**")
            engineered_features = [
                ("engagement_score", "Combined likes + replies"),
                ("high_engagement", "Top 25% engagement"),
                ("word_count", "Number of words"),
                ("sentiment_magnitude", "Absolute sentiment strength"),
                ("is_positive/negative/neutral", "Binary sentiment flags"),
                ("year/month/day_of_week", "Temporal features")
            ]
            
            for feature, description in engineered_features:
                if feature in df.columns or any(f in df.columns for f in feature.split('/')):
                    st.markdown(f"• **{feature}**: {description}")
    
    # Analysis Pipeline
    st.subheader("🔄 Analysis Pipeline")
    
    pipeline_steps = [
        ("1️⃣ Data Collection", "Scraping from Reddit & YouTube APIs"),
        ("2️⃣ Data Cleaning", "Harmonization and text preprocessing"),
        ("3️⃣ Feature Engineering", "Temporal, engagement, text features"),
        ("4️⃣ EDA & Visualization", "Interactive dashboards and insights"),
        ("5️⃣ Model Training", "Traditional ML and BERT models"),
        ("6️⃣ Evaluation", "Performance metrics and comparison")
    ]
    
    for step, description in pipeline_steps:
        st.markdown(f"**{step}** — {description}")

# ===== DATA EXPLORER PAGE =====
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("### Browse and filter the AI sentiment dataset")
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available. Please check data loading.")
        st.stop()
    
    # Search functionality
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search in comments:", placeholder="Type keywords (AI, ethics, future, etc.)")
    with col2:
        sample_size = st.slider("Sample size", 10, 100, 50)
    
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
            ("Avg Length", search_df['text_length'].mean()),
            ("Engagement Score", search_df['engagement_score'].mean())
        ]
        
        for col, (label, value) in zip(stats_cols, metrics):
            with col:
                if isinstance(value, float):
                    st.metric(label, f"{value:.2f}")
                else:
                    st.metric(label, value)
        
        # Detailed statistics
        if st.checkbox("Show detailed statistics"):
            st.dataframe(search_df.describe(), use_container_width=True)
        
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
    
    # Tab layout for different EDA sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📈 Time Analysis", "🌐 Platform Comparison", "🔍 Text Analysis"])
    
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution by platform
            if 'source' in filtered_df.columns and 'sentiment_category' in filtered_df.columns:
                sentiment_by_platform = pd.crosstab(filtered_df['source'], filtered_df['sentiment_category'])
                fig = px.bar(
                    sentiment_by_platform,
                    barmode='group',
                    title='Sentiment Distribution by Platform',
                    color_discrete_map={
                        'positive': COLORS['positive'],
                        'neutral': COLORS['neutral'],
                        'negative': COLORS['negative']
                    }
                )
                fig.update_layout(height=400, xaxis_title="Platform", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
            # Text length distribution
            fig = px.histogram(
                filtered_df,
                x='text_length',
                nbins=50,
                title='Comment Length Distribution',
                color='source' if 'source' in filtered_df.columns else None,
                color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
            )
            fig.update_layout(height=400, xaxis_title="Text Length (words)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment polarity distribution
            fig = px.histogram(
                filtered_df,
                x='sentiment_polarity',
                nbins=50,
                title='Sentiment Polarity Distribution',
                color='source' if 'source' in filtered_df.columns else None,
                marginal="box",
                color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Neutral")
            fig.update_layout(height=400, xaxis_title="Sentiment Polarity", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            # Engagement score distribution
            fig = px.histogram(
                filtered_df[filtered_df['engagement_score'] < filtered_df['engagement_score'].quantile(0.95)],
                x='engagement_score',
                nbins=50,
                title='Engagement Score Distribution (95th percentile)',
                color='source' if 'source' in filtered_df.columns else None,
                color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
            )
            fig.update_layout(height=400, xaxis_title="Engagement Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Posts over time
            if 'created_at' in filtered_df.columns:
                filtered_df['date'] = filtered_df['created_at'].dt.date
                posts_over_time = filtered_df.groupby(['date', 'source']).size().reset_index(name='count')
                
                fig = px.line(
                    posts_over_time,
                    x='date',
                    y='count',
                    color='source',
                    title='Comments Over Time by Platform',
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Number of Comments")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment over time
            if 'created_at' in filtered_df.columns:
                filtered_df['month_year'] = filtered_df['created_at'].dt.to_period('M').astype(str)
                sentiment_over_time = filtered_df.groupby(['month_year', 'source'])['sentiment_polarity'].mean().reset_index()
                
                fig = px.line(
                    sentiment_over_time,
                    x='month_year',
                    y='sentiment_polarity',
                    color='source',
                    title='Average Sentiment Over Time',
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.update_layout(height=400, xaxis_title="Month", yaxis_title="Average Sentiment")
                st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of activity by hour and day
        if 'created_at' in filtered_df.columns and 'day_of_week' in filtered_df.columns:
            filtered_df['hour'] = filtered_df['created_at'].dt.hour
            heatmap_data = filtered_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
            
            fig = px.density_heatmap(
                heatmap_data,
                x='hour',
                y='day_of_week',
                z='count',
                title='Activity Heatmap: Day of Week vs Hour',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
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
                if metric_col in filtered_df.columns:
                    for platform in filtered_df['source'].unique():
                        platform_data = filtered_df[filtered_df['source'] == platform]
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
                    title='Platform Comparison',
                    color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto',
                title='Feature Correlation Heatmap',
                zmin=-1,
                zmax=1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Text Analysis")
        
        # Word clouds
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Most Frequent Words")
            
            # Combine all text
            all_text = ' '.join(filtered_df['text_clean'].astype(str).fillna(''))
            
            if all_text.strip():
                # Get word frequencies
                words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
                word_freq = Counter(words)
                
                # Remove common words
                stop_words = set(['the', 'and', 'for', 'that', 'this', 'with', 'have', 'from', 'they', 'what'])
                for word in stop_words:
                    word_freq.pop(word, None)
                
                # Create bar chart of top words
                top_words = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
                
                fig = px.bar(
                    top_words,
                    x='Frequency',
                    y='Word',
                    orientation='h',
                    title='Top 20 Most Frequent Words',
                    color='Frequency',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Sentiment vs Engagement")
            
            # Scatter plot
            sample_df = filtered_df.sample(min(500, len(filtered_df)))
            
            fig = px.scatter(
                sample_df,
                x='sentiment_polarity',
                y='engagement_score',
                color='source',
                size='text_length',
                hover_data=['text_clean'],
                title='Sentiment vs Engagement',
                color_discrete_map={'reddit': COLORS['reddit'], 'youtube': COLORS['youtube']}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Word clouds by sentiment
        st.markdown("#### Word Clouds by Sentiment")
        
        sentiment_cols = st.columns(3)
        sentiments = ['positive', 'neutral', 'negative']
        
        for col, sentiment in zip(sentiment_cols, sentiments):
            with col:
                sentiment_text = ' '.join(
                    filtered_df[filtered_df['sentiment_category'] == sentiment]['text_clean'].astype(str).fillna('')
                )
                
                if sentiment_text.strip():
                    wordcloud = WordCloud(
                        width=400, 
                        height=300, 
                        background_color='white',
                        colormap='viridis' if sentiment == 'positive' else 'cool' if sentiment == 'neutral' else 'autumn'
                    ).generate(sentiment_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'{sentiment.title()} Sentiment')
                    st.pyplot(fig)

# ===== TRADITIONAL ML MODELS PAGE =====
elif page == "🤖 Traditional ML Models":
    st.title("🤖 Traditional Machine Learning Models")
    st.markdown("### TF-IDF + Traditional ML algorithms for sentiment prediction")
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available. Please check data loading.")
        st.stop()
    
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
                target_df = filtered_df[target_variable]
            else:
                is_binary = False
                # Convert sentiment_category to numeric labels
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                target_df = pd.Series(le.fit_transform(filtered_df[target_variable]))
        
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
                n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
                max_depth = st.selectbox("Max Depth", [None, 5, 10, 20, 30])
    
    # Prepare data
    st.subheader("📊 Data Preparation")
    
    # Define features
    features = ["text_tfidf"]
    if use_numeric:
        numeric_features = ['text_length', 'sentiment_subjectivity', 'word_count']
        features.extend([f for f in numeric_features if f in filtered_df.columns])
    
    X = filtered_df[features]
    y = target_df
    
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
        max_df=0.90,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    transformers.append(("tfidf", tfidf, "text_tfidf"))
    
    if use_numeric and len(features) > 1:
        numeric_cols = [f for f in features if f != "text_tfidf"]
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    
    preprocess = ColumnTransformer(transformers=transformers)
    
    # Train model
    st.subheader("🚀 Model Training")
    
    with st.spinner("Training model..."):
        if model_choice == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        elif model_choice == "Linear SVC":
            clf = LinearSVC(class_weight="balanced", max_iter=5000, C=0.5)
        elif model_choice == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=random_state
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
            roc_auc = roc_auc_score(y_test, y_pred_proba)
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
    
    # Feature importance (for interpretable models)
    if model_choice == "Logistic Regression":
        st.subheader("🔍 Feature Importance")
        
        try:
            # Get feature names
            feature_names = []
            
            # Get TF-IDF feature names
            tfidf_features = model.named_steps['preprocess'].named_transformers_['tfidf'].get_feature_names_out()
            feature_names.extend([f"word_{f}" for f in tfidf_features])
            
            if use_numeric:
                numeric_features = [f for f in features if f != "text_tfidf"]
                feature_names.extend(numeric_features)
            
            # Get coefficients
            if hasattr(model.named_steps['clf'], 'coef_'):
                coefficients = model.named_steps['clf'].coef_[0]
                
                # Create dataframe
                feature_importance = pd.DataFrame({
                    'Feature': feature_names[:len(coefficients)],
                    'Coefficient': coefficients
                })
                
                # Show top features
                top_n = st.slider("Number of top features to show", 5, 30, 10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    top_positive = feature_importance.nlargest(top_n, 'Coefficient')
                    st.markdown(f"**Top {top_n} Positive Features**")
                    fig = px.bar(
                        top_positive,
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        color='Coefficient',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    top_negative = feature_importance.nsmallest(top_n, 'Coefficient')
                    st.markdown(f"**Top {top_n} Negative Features**")
                    fig = px.bar(
                        top_negative,
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        color='Coefficient',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.warning(f"Could not extract feature importance: {str(e)}")
    
    # Sample predictions
    st.subheader("🔮 Sample Predictions")
    
    sample_size = min(10, len(X_test))
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
            'Text': X_test.iloc[idx]['text_tfidf'][:100] + '...',
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
                target_df = filtered_df[target_variable]
            else:
                is_binary = False
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                target_df = pd.Series(le.fit_transform(filtered_df[target_variable]))
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="bert_test")
            random_state = st.number_input("Random State", 1, 100, 42, key="bert_random")
            
            use_numeric_bert = st.checkbox("Include Numeric Features", value=True)
        
        with col3:
            bert_classifier = st.selectbox(
                "Classifier",
                ["Logistic Regression", "Random Forest", "Gradient Boosting"]
            )
    
    # Prepare data
    st.subheader("📊 Data Preparation")
    
    st.info("**Note**: Computing BERT embeddings may take some time...")
    
    # Compute BERT embeddings
    with st.spinner("Computing BERT embeddings..."):
        try:
            # Use BERT-optimized text
            texts = filtered_df['text_bert'].fillna('').tolist()
            embeddings = bert_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
            if use_numeric_bert:
                numeric_features = ['text_length', 'sentiment_subjectivity', 'word_count']
                numeric_data = filtered_df[[f for f in numeric_features if f in filtered_df.columns]].values
                
                if numeric_data.size > 0:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    numeric_scaled = scaler.fit_transform(numeric_data)
                    X = np.hstack([embeddings, numeric_scaled])
                else:
                    X = embeddings
            else:
                X = embeddings
            
            y = target_df.values if hasattr(target_df, 'values') else target_df
            
            st.success(f"✅ BERT embeddings computed: {X.shape[0]} samples, {X.shape[1]} features")
            
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
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        elif bert_classifier == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                random_state=random_state
            )
        elif bert_classifier == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        
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
            roc_auc = roc_auc_score(y_test, y_pred_proba)
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
    
    # Feature importance (for tree-based models)
    if bert_classifier in ["Random Forest", "Gradient Boosting"]:
        st.subheader("🔍 Feature Importance")
        
        try:
            if hasattr(clf, 'feature_importances_'):
                n_features = min(20, X.shape[1])
                indices = np.argsort(clf.feature_importances_)[::-1][:n_features]
                
                # Create labels
                labels = [f"BERT_feature_{i}" for i in range(embeddings.shape[1])]
                if use_numeric_bert:
                    numeric_labels = ['text_length', 'sentiment_subjectivity', 'word_count'][:X.shape[1]-embeddings.shape[1]]
                    labels.extend(numeric_labels)
                
                importance_df = pd.DataFrame({
                    'Feature': [labels[i] for i in indices],
                    'Importance': clf.feature_importances_[indices]
                })
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    title=f'Top {n_features} Feature Importances'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Could not extract feature importance: {str(e)}")

# ===== MODEL COMPARISON PAGE =====
elif page == "📋 Model Comparison":
    st.title("📋 Model Comparison")
    st.markdown("### Benchmarking different modeling approaches")
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available. Please check data loading.")
        st.stop()
    
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
                y = filtered_df[target_variable]
            else:
                is_binary = False
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(filtered_df[target_variable]))
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="compare_test")
            random_state = st.number_input("Random State", 1, 100, 42, key="compare_random")
            
            models_to_compare = st.multiselect(
                "Models to Compare",
                ["TF-IDF + Logistic Regression", "TF-IDF + Random Forest", 
                 "BERT + Logistic Regression", "BERT + Random Forest"],
                default=["TF-IDF + Logistic Regression", "BERT + Logistic Regression"]
            )
    
    # Train and compare models
    st.subheader("📊 Model Training & Comparison")
    
    results = []
    
    # Prepare TF-IDF features
    if any("TF-IDF" in model for model in models_to_compare):
        with st.spinner("Preparing TF-IDF features..."):
            tfidf = TfidfVectorizer(
                min_df=2,
                max_df=0.90,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True
            )
            X_tfidf = tfidf.fit_transform(filtered_df['text_tfidf'])
            
            # Split data
            X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=test_size, random_state=random_state, stratify=y if is_binary else None
            )
    
    # Prepare BERT features
    if any("BERT" in model for model in models_to_compare):
        with st.spinner("Computing BERT embeddings..."):
            if bert_model is not None:
                texts = filtered_df['text_bert'].fillna('').tolist()
                X_bert = bert_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                
                # Split data
                X_train_bert, X_test_bert, y_train, y_test = train_test_split(
                    X_bert, y, test_size=test_size, random_state=random_state, stratify=y if is_binary else None
                )
    
    # Train each model
    with st.spinner("Training models..."):
        for model_name in models_to_compare:
            with st.spinner(f"Training {model_name}..."):
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
                    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
                
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