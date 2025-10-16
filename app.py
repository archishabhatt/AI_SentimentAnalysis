import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Page config
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
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
    }
    h1 {
        color: #2c3e50;
        padding-bottom: 10px;
    }
    h2 {
        color: #34495e;
        padding-top: 20px;
    }
    h3 {
        color: #546e7a;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/reddit_ai_cleaned.csv')
    df['created_datetime'] = pd.to_datetime(df['created_datetime'])
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Sidebar
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Section:",
    ["📖 Data Overview", "📈 Sentiment Analysis", "💬 Engagement Patterns", "☁️ Text Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

# Subreddit filter
subreddits = st.sidebar.multiselect(
    "Subreddits",
    options=sorted(df['subreddit'].unique()),
    default=df['subreddit'].unique()
)

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters
if len(date_range) == 2:
    filtered_df = df[
        (df['subreddit'].isin(subreddits)) &
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1])
    ]
else:
    filtered_df = df[df['subreddit'].isin(subreddits)]

st.sidebar.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} posts**")

# Color scheme
COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'purple': '#9b59b6',
    'teal': '#1abc9c'
}

# ===== DATA OVERVIEW PAGE =====
if page == "📖 Data Overview":
    st.title("🤖 AI Sentiment Analysis on Reddit")
    st.markdown("### Understanding sentiment trends in AI discussions across Reddit communities")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Posts", f"{len(filtered_df):,}")
    with col2:
        st.metric("Avg Score", f"{filtered_df['score'].mean():.1f}")
    with col3:
        st.metric("Avg Comments", f"{filtered_df['num_comments'].mean():.1f}")
    with col4:
        st.metric("Avg Sentiment", f"{filtered_df['sentiment_polarity'].mean():.3f}")
    with col5:
        st.metric("Subreddits", len(filtered_df['subreddit'].unique()))
    
    st.markdown("")
    
    # Dataset description
    st.subheader("About the Dataset")
    st.markdown("""
    This dataset was scraped using the inbuilt Reddit API via PRAW. It includes posts from various AI-related subreddits, filtered to focus on discussions that are opinionated and touch on ethical or societal implications of AI. The sentiment of each post was analyzed using TextBlob, providing insights into the polarity and subjectivity of the content.
    The original 13 features include:
    - **subreddit**: The subreddit where the post was made
    - **title**: The title of the post (cleaned for analysis)
    - **selftext**: The body text of the post (cleaned for analysis)
    - **score**: The post's score (upvotes - downvotes)
    - **upvote_ratio**: The ratio of upvotes to total votes (0 to 1)
    - **id**: The unique identifier for the post (removed durning cleaning)
    - **url**: The URL of the post (removed during cleaning)
    - **created_utc**: The UTC timestamp when the post was created (converted to datetime during cleaning))
    - **num_comments**: Number of comments on the post
    - **post_length**: Number of words in the full post (title + selftext)
    - **title_length**: Number of words in the title
    - **sentiment_polarity**: Sentiment polarity score (-1 {very negative} to 1 {very positive})
    - **sentiment_subjectivity**: Sentiment subjectivity score (0 {very objective/factual} to 1 {very opinionated})
    """)
    
    st.markdown("")
    
    # Posts by subreddit
    st.subheader("Post Distribution Across Subreddits")
    subreddit_counts = filtered_df['subreddit'].value_counts()
    fig = px.bar(
        x=subreddit_counts.values,
        y=subreddit_counts.index,
        orientation='h',
        labels={'x': 'Number of Posts', 'y': 'Subreddit'},
        color=subreddit_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    summary_stats = filtered_df[['score', 'num_comments', 'upvote_ratio', 
                                   'sentiment_polarity', 'sentiment_subjectivity']].describe()
    st.dataframe(summary_stats.style.format("{:.2f}"), use_container_width=True)

# ===== SENTIMENT ANALYSIS PAGE =====
elif page == "📈 Sentiment Analysis":
    st.title("📈 Sentiment Analysis")
    st.markdown("### Deep dive into sentiment patterns and trends")
    st.markdown("---")
    
    # Sentiment trends over time
    st.subheader("Sentiment Trends Over Time")
    st.markdown("*Tracking how sentiment polarity and subjectivity evolve across the timeline*")
    
    daily_sentiment = filtered_df.groupby('date').agg({
        'sentiment_polarity': 'mean',
        'sentiment_subjectivity': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['sentiment_polarity'],
        mode='lines+markers',
        name='Polarity',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['sentiment_subjectivity'],
        mode='lines+markers',
        name='Subjectivity',
        line=dict(color=COLORS['warning'], width=2),
        marker=dict(size=6)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, 
                  annotation_text="Neutral")
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        hovermode='x unified',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("")
    
    # Sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Polarity Distribution")
        fig = px.histogram(
            filtered_df,
            x='sentiment_polarity',
            nbins=50,
            color_discrete_sequence=[COLORS['primary']]
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="Neutral", annotation_position="top")
        fig.update_layout(
            xaxis_title='Sentiment Polarity',
            yaxis_title='Count',
            showlegend=False,
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        positive_pct = (filtered_df['sentiment_polarity'] > 0).mean() * 100
        negative_pct = (filtered_df['sentiment_polarity'] < 0).mean() * 100
        neutral_pct = (filtered_df['sentiment_polarity'] == 0).mean() * 100
        st.info(f"**Distribution**: {positive_pct:.1f}% Positive | {negative_pct:.1f}% Negative | {neutral_pct:.1f}% Neutral")
    
    with col2:
        st.subheader("Sentiment Subjectivity Distribution")
        fig = px.histogram(
            filtered_df,
            x='sentiment_subjectivity',
            nbins=50,
            color_discrete_sequence=[COLORS['purple']]
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="blue", 
                     annotation_text="Mid-point", annotation_position="top")
        fig.update_layout(
            xaxis_title='Sentiment Subjectivity',
            yaxis_title='Count',
            showlegend=False,
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        avg_subj = filtered_df['sentiment_subjectivity'].mean()
        st.info(f"**Average Subjectivity**: {avg_subj:.3f} - Posts tend to be {'more subjective' if avg_subj > 0.5 else 'more objective'}")
    
    st.markdown("")
    
    # Polarity vs Subjectivity scatter
    st.subheader("Polarity vs Subjectivity Relationship")
    st.markdown("*Size represents number of comments, color represents post score*")
    
    sample_size = min(1000, len(filtered_df))
    sample_df = filtered_df.sample(sample_size)
    
    fig = px.scatter(
        sample_df,
        x='sentiment_polarity',
        y='sentiment_subjectivity',
        color='score',
        size='num_comments',
        hover_data=['title_clean', 'subreddit'],
        color_continuous_scale='Viridis'
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=0.5, line_dash="dash", line_color="blue", opacity=0.5)
    fig.update_layout(
        xaxis_title='Sentiment Polarity (Negative ← → Positive)',
        yaxis_title='Sentiment Subjectivity (Objective ← → Subjective)',
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("")
    
    # Sentiment by subreddit
    st.subheader("Average Sentiment by Subreddit")
    subreddit_sentiment = filtered_df.groupby('subreddit')['sentiment_polarity'].mean().sort_values()
    
    fig = px.bar(
        x=subreddit_sentiment.values,
        y=subreddit_sentiment.index,
        orientation='h',
        color=subreddit_sentiment.values,
        color_continuous_scale='RdYlGn',
        labels={'x': 'Average Sentiment Polarity', 'y': 'Subreddit'}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== ENGAGEMENT PATTERNS PAGE =====
elif page == "💬 Engagement Patterns":
    st.title("💬 Engagement Patterns")
    st.markdown("### Exploring what drives user engagement")
    st.markdown("---")
    
    # Post volume over time
    st.subheader("Post Volume Over Time")
    st.markdown("*Understanding activity patterns across the timeline*")
    
    posts_per_day = filtered_df.groupby('date').size().reset_index(name='count')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=posts_per_day['date'],
        y=posts_per_day['count'],
        mode='lines',
        fill='tozeroy',
        line=dict(color=COLORS['teal'], width=2),
        fillcolor='rgba(26, 188, 156, 0.2)'
    ))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Posts',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Peak activity
    peak_day = posts_per_day.loc[posts_per_day['count'].idxmax()]
    st.info(f"📈 **Peak Activity**: {peak_day['count']} posts on {peak_day['date'].strftime('%B %d, %Y')}")
    
    st.markdown("")
    
    # Engagement correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score vs Comments")
        st.markdown("*Do higher-scored posts get more comments?*")
        
        fig = px.scatter(
            filtered_df,
            x='score',
            y='num_comments',
            trendline='ols',
            color='sentiment_polarity',
            color_continuous_scale='RdYlGn',
            opacity=0.6
        )
        fig.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation
        corr = filtered_df[['score', 'num_comments']].corr().iloc[0, 1]
        st.metric("Correlation", f"{corr:.3f}")
    
    with col2:
        st.subheader("Post Length vs Score")
        st.markdown("*Does post length affect engagement?*")
        
        fig = px.scatter(
            filtered_df,
            x='post_length',
            y='score',
            trendline='ols',
            color='sentiment_polarity',
            color_continuous_scale='RdYlGn',
            opacity=0.6
        )
        fig.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation
        corr = filtered_df[['post_length', 'score']].corr().iloc[0, 1]
        st.metric("Correlation", f"{corr:.3f}")
    
    st.markdown("")
    
    # Engagement metrics by subreddit
    st.subheader("Engagement Metrics by Subreddit")
    engagement_by_sub = filtered_df.groupby('subreddit').agg({
        'score': 'mean',
        'num_comments': 'mean',
        'upvote_ratio': 'mean',
        'post_length': 'mean'
    }).round(2)
    engagement_by_sub.columns = ['Avg Score', 'Avg Comments', 'Avg Upvote Ratio', 'Avg Post Length']
    st.dataframe(engagement_by_sub.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    st.markdown("")
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    st.markdown("*Understanding relationships between different metrics*")
    
    corr_cols = ['score', 'num_comments', 'upvote_ratio', 'post_length', 
                 'sentiment_polarity', 'sentiment_subjectivity']
    corr_matrix = filtered_df[corr_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== TEXT ANALYSIS PAGE =====
elif page == "☁️ Text Analysis":
    st.title("☁️ Text Analysis")
    st.markdown("### Exploring the language of AI discussions")
    st.markdown("---")
    
    # Word clouds
    st.subheader("Word Clouds by Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Positive Sentiment Posts")
        st.caption("Posts with polarity > 0.1")
        positive_text = ' '.join(filtered_df[filtered_df['sentiment_polarity'] > 0.1]['title_clean'].astype(str))
        
        if positive_text.strip():
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='Greens',
                max_words=50,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(positive_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig)
        else:
            st.info("No positive posts in current selection")
    
    with col2:
        st.markdown("#### Negative Sentiment Posts")
        st.caption("Posts with polarity < -0.1")
        negative_text = ' '.join(filtered_df[filtered_df['sentiment_polarity'] < -0.1]['title_clean'].astype(str))
        
        if negative_text.strip():
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='Reds',
                max_words=50,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(negative_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig)
        else:
            st.info("No negative posts in current selection")
    
    st.markdown("")
    st.markdown("---")
    
    # Most common words
    st.subheader("Most Common Words in Post Titles")
    
    all_words = ' '.join(filtered_df['title_clean'].astype(str)).lower().split()
    stop_words = {'i','you','we','the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'they', 'them', 'their',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'how', 'why',
                  'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'like',
                  'what', 'from', 'about', 'where', 'when', 'your', 'my', 'so', 'if', 'as', 'it', 'its'}
    filtered_words = [word for word in all_words if word not in stop_words and len(word) > 3]
    word_counts = Counter(filtered_words).most_common(25)
    
    if word_counts:
        words, counts = zip(*word_counts)
        fig = px.bar(
            x=counts,
            y=words,
            orientation='h',
            labels={'x': 'Frequency', 'y': 'Word'},
            color=counts,
            color_continuous_scale='Teal'
        )
        fig.update_layout(
            height=700,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:** Reddit API")
st.sidebar.caption(f"Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")