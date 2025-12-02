import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import umap.umap_ as umap
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import seaborn as sns
from sentence_transformers import SentenceTransformer
import joblib

# Page config
st.set_page_config(
    page_title="AI Sentiment & Engagement Analysis",
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
    .model-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .performance-metrics {
        display: flex;
        gap: 20px;
        margin-top: 20px;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/reddit_ai_cleaned.csv')
    df['created_datetime'] = pd.to_datetime(df['created_datetime'])
    df['date'] = pd.to_datetime(df['date'])
    df['high_score'] = (df['score'] >= df['score'].median()).astype(int)
    return df

df = load_data()

# Load BERT embeddings (cached for performance)
@st.cache_resource
def load_bert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def compute_embeddings(_texts):
    model = load_bert_model()
    return model.encode(_texts.tolist(), show_progress_bar=False)

# Sidebar
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Section:",
    ["📖 Data Overview", "📈 EDA/IDA", "🤖 Traditional Models", "🧠 BERT Models"]
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
    st.title("🤖 AI Sentiment & Engagement Analysis on Reddit")
    st.markdown("### Comprehensive analysis of AI discussions including engagement prediction modeling")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Posts", f"{len(filtered_df):,}")
    with col2:
        st.metric("Avg Score", f"{filtered_df['score'].mean():.1f}")
    with col3:
        st.metric("Avg Comments", f"{filtered_df['num_comments'].mean():.1f}")
    with col4:
        st.metric("Avg Sentiment", f"{filtered_df['sentiment_polarity'].mean():.3f}")
    with col5:
        st.metric("High Score Posts", f"{(filtered_df['high_score'] == 1).sum():,}")
    with col6:
        st.metric("Target Balance", f"{filtered_df['high_score'].mean():.1%}")
    
    st.markdown("")
    
    # Dataset description
    st.subheader("About the Dataset")
    st.markdown("""
    This dataset was scraped using the inbuilt Reddit API via PRAW. It includes posts from various AI-related subreddits, 
    filtered to focus on discussions that are opinionated and touch on ethical or societal implications of AI. 
    
    ### Target Variable: `high_score`
    A binary label indicating whether a post received above-median engagement (score). This is used for predicting 
    which posts will get high engagement based on text content and metadata.
    
    ### Features include:
    - **Text Features**: `title_clean`, `selftext_clean` (cleaned text for analysis)
    - **Engagement Metrics**: `score`, `num_comments`, `upvote_ratio`
    - **Content Metrics**: `post_length`, `sentiment_polarity`, `sentiment_subjectivity`
    - **Metadata**: `subreddit`, `created_datetime`, `date`, `year`
    - **Target**: `high_score` (1 = high engagement, 0 = low engagement)
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
                                   'sentiment_polarity', 'sentiment_subjectivity', 
                                   'post_length', 'high_score']].describe()
    st.dataframe(summary_stats.style.format("{:.2f}"), use_container_width=True)

# ===== EDA/IDA PAGE =====
elif page == "📈 EDA/IDA":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("### Understanding data patterns, distributions, and relationships")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📈 Time Analysis", "🔗 Correlations", "🎯 Target Analysis"])
    
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig = px.histogram(
                filtered_df,
                x='score',
                nbins=50,
                title='Score Distribution',
                color_discrete_sequence=[COLORS['primary']]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment polarity distribution
            fig = px.histogram(
                filtered_df,
                x='sentiment_polarity',
                nbins=50,
                title='Sentiment Polarity Distribution',
                color_discrete_sequence=[COLORS['success']]
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Comments distribution
            fig = px.histogram(
                filtered_df,
                x='num_comments',
                nbins=50,
                title='Comments Distribution',
                color_discrete_sequence=[COLORS['warning']]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Post length distribution
            fig = px.histogram(
                filtered_df,
                x='post_length',
                nbins=50,
                title='Post Length Distribution',
                color_discrete_sequence=[COLORS['purple']]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Temporal Analysis")
        
        # Posts over time
        posts_over_time = filtered_df.groupby('date').size().reset_index(name='count')
        fig = px.line(
            posts_over_time,
            x='date',
            y='count',
            title='Posts Over Time',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Engagement over time
        engagement_over_time = filtered_df.groupby('date').agg({
            'score': 'mean',
            'num_comments': 'mean',
            'high_score': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                engagement_over_time,
                x='date',
                y='score',
                title='Average Score Over Time',
                markers=True
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                engagement_over_time,
                x='date',
                y='high_score',
                title='High Score Ratio Over Time',
                markers=True
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Correlation heatmap
        corr_cols = ['score', 'num_comments', 'upvote_ratio', 'post_length',
                     'sentiment_polarity', 'sentiment_subjectivity', 'high_score']
        corr_matrix = filtered_df[corr_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Feature Correlation Heatmap',
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        st.subheader("Feature Relationships")
        
        scatter_cols = st.multiselect(
            "Select features for scatter plot:",
            options=corr_cols,
            default=['score', 'num_comments']
        )
        
        if len(scatter_cols) >= 2:
            fig = px.scatter(
                filtered_df,
                x=scatter_cols[0],
                y=scatter_cols[1],
                color='high_score',
                title=f'{scatter_cols[0]} vs {scatter_cols[1]}',
                opacity=0.6
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Target Variable Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target distribution
            target_counts = filtered_df['high_score'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=['Low Engagement', 'High Engagement'],
                title='Target Distribution',
                color_discrete_sequence=[COLORS['secondary'], COLORS['success']]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance by target
            features_to_compare = ['post_length', 'sentiment_polarity', 'sentiment_subjectivity']
            comparison_data = []
            
            for feature in features_to_compare:
                high_mean = filtered_df[filtered_df['high_score'] == 1][feature].mean()
                low_mean = filtered_df[filtered_df['high_score'] == 0][feature].mean()
                comparison_data.append({
                    'Feature': feature,
                    'High Engagement': high_mean,
                    'Low Engagement': low_mean
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.set_index('Feature')
            
            fig = px.bar(
                comparison_df,
                barmode='group',
                title='Feature Means by Engagement Level'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Word clouds by engagement level
        st.subheader("Text Analysis by Engagement Level")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### High Engagement Posts")
            high_text = ' '.join(filtered_df[filtered_df['high_score'] == 1]['title_clean'].astype(str))
            if high_text.strip():
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(high_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        with col2:
            st.markdown("#### Low Engagement Posts")
            low_text = ' '.join(filtered_df[filtered_df['high_score'] == 0]['title_clean'].astype(str))
            if low_text.strip():
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(low_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

# ===== TRADITIONAL MODELS PAGE =====
elif page == "🤖 Traditional Models":
    st.title("🤖 Traditional ML Models for Engagement Prediction")
    st.markdown("### Using TF-IDF and traditional machine learning algorithms")
    st.markdown("---")
    
    with st.expander("📋 Model Setup & Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State", 1, 100, 42)
        
        with col2:
            use_numeric = st.checkbox("Include Numeric Features", value=True)
            use_titles = st.checkbox("Use Title Text", value=True)
            use_selftext = st.checkbox("Use Self Text", value=True)
        
        with col3:
            model_choice = st.selectbox(
                "Select Model",
                ["Logistic Regression", "Linear SVC", "Random Forest"]
            )
            
            if model_choice == "Random Forest":
                n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
                max_depth = st.selectbox("Max Depth", [None, 5, 10, 20, 30])
    
    # Prepare data
    st.subheader("Data Preparation")
    
    # Define features based on selection
    features = []
    if use_titles:
        features.append("title_clean")
    if use_selftext:
        features.append("selftext_clean")
    if use_numeric:
        features.extend(["post_length", "sentiment_polarity", "sentiment_subjectivity"])
    
    X = filtered_df[features]
    y = filtered_df['high_score']
    
    st.info(f"**Features Used**: {', '.join(features)}")
    st.info(f"**Dataset Size**: {len(X)} posts | **Positive Class**: {y.mean():.1%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create preprocessing pipeline
    transformers = []
    
    if use_titles or use_selftext:
        tfidf = TfidfVectorizer(
            min_df=2,
            max_df=0.90,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        
        if use_titles:
            transformers.append(("tfidf_title", tfidf, "title_clean"))
        if use_selftext:
            transformers.append(("tfidf_text", tfidf, "selftext_clean"))
    
    if use_numeric:
        numeric_features = ["post_length", "sentiment_polarity", "sentiment_subjectivity"]
        transformers.append(("numeric", StandardScaler(), numeric_features))
    
    preprocess = ColumnTransformer(transformers=transformers)
    
    # Train model
    st.subheader("Model Training")
    
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
        f1 = f1_score(y_test, y_pred)
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = "N/A"
    
    # Display results
    st.subheader("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("F1 Score", f"{f1:.3f}")
    with col3:
        st.metric("ROC AUC", f"{roc_auc:.3f}" if isinstance(roc_auc, (int, float)) else roc_auc)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Low', 'Predicted High'],
                yticklabels=['Actual Low', 'Actual High'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # Feature importance (for interpretable models)
    if model_choice == "Logistic Regression":
        st.subheader("Top Feature Coefficients")
        
        try:
            # Get feature names from the pipeline
            feature_names = []
            
            if use_titles:
                title_features = model.named_steps['preprocess'].named_transformers_['tfidf_title'].get_feature_names_out()
                feature_names.extend([f"title_{f}" for f in title_features])
            
            if use_selftext:
                text_features = model.named_steps['preprocess'].named_transformers_['tfidf_text'].get_feature_names_out()
                feature_names.extend([f"text_{f}" for f in text_features])
            
            if use_numeric:
                feature_names.extend(["post_length", "sentiment_polarity", "sentiment_subjectivity"])
            
            # Get coefficients
            coefficients = model.named_steps['clf'].coef_[0]
            
            # Create dataframe
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            
            # Show top positive and negative features
            top_positive = feature_importance.nlargest(10, 'Coefficient')
            top_negative = feature_importance.nsmallest(10, 'Coefficient')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Positive Features**")
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
                st.markdown("**Top Negative Features**")
                fig = px.bar(
                    top_negative,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    color='Coefficient',
                    color_continuous_scale='Reds_r'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Could not extract feature importance: {str(e)}")
    
    # Sample predictions
    st.subheader("Sample Predictions")
    
    sample_size = min(10, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    
    sample_data = []
    for idx in sample_indices:
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        confidence = y_pred_proba[idx] if y_pred_proba is not None else "N/A"
        
        sample_data.append({
            'Title': X_test.iloc[idx]['title_clean'][:100] + '...' if use_titles else 'N/A',
            'Actual': 'High' if actual == 1 else 'Low',
            'Predicted': 'High' if predicted == 1 else 'Low',
            'Confidence': f"{confidence:.3f}" if isinstance(confidence, (int, float)) else confidence,
            'Correct': '✓' if actual == predicted else '✗'
        })
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)

# ===== BERT MODELS PAGE =====
elif page == "🧠 BERT Models":
    st.title("🧠 BERT-Based Models for Engagement Prediction")
    st.markdown("### Using sentence embeddings for enhanced text understanding")
    st.markdown("---")
    
    with st.expander("📋 BERT Model Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="bert_test")
            random_state = st.number_input("Random State", 1, 100, 42, key="bert_random")
        
        with col2:
            bert_model = st.selectbox(
                "BERT Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]
            )
            
            use_numeric_bert = st.checkbox("Include Numeric Features with BERT", value=True)
        
        with col3:
            bert_classifier = st.selectbox(
                "Classifier",
                ["Logistic Regression", "Random Forest", "Neural Network"]
            )
    
    # Prepare data
    st.subheader("Data Preparation")
    
    st.info("**Note**: Computing BERT embeddings may take some time for large datasets...")
    
    # Compute BERT embeddings
    with st.spinner("Computing BERT embeddings..."):
        embeddings = compute_embeddings(filtered_df['text'])
        
        if use_numeric_bert:
            numeric_features = filtered_df[['post_length', 'sentiment_polarity', 'sentiment_subjectivity']].values
            X = np.hstack([embeddings, numeric_features])
        else:
            X = embeddings
    
    y = filtered_df['high_score'].values
    
    st.success(f"✅ BERT embeddings computed: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train model
    st.subheader("Model Training")
    
    with st.spinner(f"Training {bert_classifier}..."):
        if bert_classifier == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        elif bert_classifier == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                random_state=random_state
            )
        elif bert_classifier == "Neural Network":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=random_state
            )
        
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"
    
    # Display results
    st.subheader("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("F1 Score", f"{f1:.3f}")
    with col3:
        st.metric("ROC AUC", f"{roc_auc:.3f}" if isinstance(roc_auc, (int, float)) else roc_auc)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=['Predicted Low', 'Predicted High'],
                yticklabels=['Actual Low', 'Actual High'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # UMAP visualization
    st.subheader("UMAP Visualization of BERT Embeddings")
    
    with st.spinner("Creating UMAP visualization..."):
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=random_state)
        embedding_2d = reducer.fit_transform(embeddings[:1000])  # Limit to 1000 points for speed
        
        # Create scatter plot colored by engagement
        viz_df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'engagement': filtered_df['high_score'].iloc[:1000].map({0: 'Low', 1: 'High'})
        })
        
        fig = px.scatter(
            viz_df,
            x='x',
            y='y',
            color='engagement',
            title='UMAP Projection of BERT Embeddings',
            color_discrete_map={'Low': COLORS['secondary'], 'High': COLORS['success']},
            opacity=0.7
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison with traditional models
    st.subheader("Model Comparison")
    
    # Train a traditional model for comparison
    with st.spinner("Training traditional model for comparison..."):
        # Traditional model using TF-IDF
        tfidf = TfidfVectorizer(
            min_df=2,
            max_df=0.90,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        
        X_trad = tfidf.fit_transform(filtered_df['text'])
        X_trad_train, X_trad_test, y_trad_train, y_trad_test = train_test_split(
            X_trad, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        trad_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        trad_clf.fit(X_trad_train, y_trad_train)
        trad_pred = trad_clf.predict(X_trad_test)
        trad_accuracy = accuracy_score(y_trad_test, trad_pred)
        trad_f1 = f1_score(y_trad_test, trad_pred)
    
    # Create comparison chart
    comparison_data = pd.DataFrame({
        'Model': ['BERT + Classifier', 'TF-IDF + Logistic Regression'],
        'Accuracy': [accuracy, trad_accuracy],
        'F1 Score': [f1, trad_f1]
    })
    
    fig = px.bar(
        comparison_data,
        x='Model',
        y=['Accuracy', 'F1 Score'],
        barmode='group',
        title='Model Comparison',
        color_discrete_sequence=[COLORS['primary'], COLORS['warning']]
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample predictions
    st.subheader("Sample Predictions with Confidence")
    
    if y_pred_proba is not None:
        sample_size = min(10, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        
        sample_data = []
        for idx in sample_indices:
            actual = y_test[idx]
            predicted = y_pred[idx]
            confidence = y_pred_proba[idx]
            
            sample_data.append({
                'Text': filtered_df.iloc[X_test.index[idx]]['title_clean'][:100] + '...',
                'Actual': 'High' if actual == 1 else 'Low',
                'Predicted': 'High' if predicted == 1 else 'Low',
                'Confidence': f"{confidence:.3f}",
                'Correct': '✓' if actual == predicted else '✗'
            })
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:** Reddit API")
st.sidebar.caption(f"Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# Add download functionality for trained models
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Export Models")

if st.sidebar.button("Save Current Model State"):
    # This would save the current model state
    st.sidebar.success("Model state saved (placeholder)")