import streamlit as st
import time
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main color scheme - sage/forest green with vibrant accents */
    :root {
        --primary-color: #4A7C59;
        --secondary-color: #6B8E7F;
        --accent-color: #8FA998;
        --background-color: #F5F7F5;
        --text-color: #2C3E2E;
        --positive-color: #3498db;
        --negative-color: #e74c3c;
        --neutral-color: #95a5a6;
    }
    
    /* Main container */
    .main {
        background-color: var(--background-color);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C5530 0%, #4A7C59 50%, #6B8E7F 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2C5530 !important;
        font-family: 'Helvetica Neue', sans-serif;
        border-bottom: 2px solid #4A7C59;
        padding-bottom: 0.5rem;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(74, 124, 89, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #4A7C59;
    }
    
    /* Performance cards */
    .perf-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(74, 124, 89, 0.1);
        border-top: 4px solid #4A7C59;
        transition: transform 0.2s;
    }
    
    .perf-card:hover {
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #4A7C59;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #8FA998;
        color: white;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4A7C59;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Caching functions
@st.cache_data
def load_data():
    """Load the final processed dataset"""
    try:
        df = pd.read_csv('data/final_df.csv')
        
        # Normalize column names
        column_mapping = {
            'sentiment_label': 'sentiment',
            'text_clean': 'cleaned_text',
            'created_at': 'timestamp'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure we have required columns
        if 'cleaned_text' not in df.columns and 'text' in df.columns:
            df['cleaned_text'] = df['text']
        
        # Handle timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['date'] = df['timestamp'].dt.date
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['month_name'] = df['timestamp'].dt.strftime('%B')
        
        # Remove any nulls in critical columns
        critical_cols = ['cleaned_text', 'sentiment']
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        # Add engineered features if they don't exist
        if 'cleaned_text' in df.columns:
            df['text_length'] = df['cleaned_text'].astype(str).apply(len)
            df['word_count'] = df['cleaned_text'].astype(str).apply(lambda x: len(x.split()))
            df['avg_word_length'] = df['cleaned_text'].astype(str).apply(
                lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
            )
            
            # AI-specific keywords
            ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 
                          'neural network', 'algorithm', 'chatgpt', 'gpt', 'llm', 'transformer']
            
            def count_ai_keywords(text):
                text_lower = str(text).lower()
                return sum(1 for keyword in ai_keywords if keyword in text_lower)
            
            df['ai_keyword_count'] = df['cleaned_text'].apply(count_ai_keywords)
            
            # Sentiment intensity indicators
            df['exclamation_count'] = df['cleaned_text'].astype(str).apply(lambda x: x.count('!'))
            df['question_count'] = df['cleaned_text'].astype(str).apply(lambda x: x.count('?'))
            df['has_question'] = df['question_count'] > 0
            df['has_exclamation'] = df['exclamation_count'] > 0
        
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/final_df.csv' exists.")
        return None

@st.cache_data
def load_model_results():
    """Load actual model results from training"""
    return {
        'logistic_regression': {
            'accuracy': 0.7454,
            'f1': 0.7461,
            'precision': 0.5733,
            'recall': 0.5633,
            'time': 3.35,
            'confusion_matrix': np.array([[121, 10, 50], 
                                          [3, 4, 11], 
                                          [33, 46, 319]]),
            'classification_report': {
                'Negative': {'precision': 0.62, 'recall': 0.67, 'f1-score': 0.64, 'support': 181},
                'Neutral': {'precision': 0.27, 'recall': 0.22, 'f1-score': 0.24, 'support': 18},
                'Positive': {'precision': 0.83, 'recall': 0.80, 'f1-score': 0.82, 'support': 398}
            }
        },
        'random_forest': {
            'accuracy': 0.6750,
            'f1': 0.6685,
            'precision': 0.6633,
            'recall': 0.4800,
            'time': 0.42,
            'confusion_matrix': np.array([[97, 15, 69], 
                                          [5, 3, 10], 
                                          [24, 60, 314]]),
            'classification_report': {
                'Negative': {'precision': 0.49, 'recall': 0.48, 'f1-score': 0.48, 'support': 181},
                'Neutral': {'precision': 0.75, 'recall': 0.17, 'f1-score': 0.27, 'support': 18},
                'Positive': {'precision': 0.75, 'recall': 0.79, 'f1-score': 0.77, 'support': 398}
            }
        },
        'svc': {
            'accuracy': 0.6382,
            'f1': 0.6438,
            'precision': 0.5867,
            'recall': 0.5133,
            'time': 0.94,
            'confusion_matrix': np.array([[159, 2, 20], 
                                          [5, 2, 11], 
                                          [179, 15, 204]]),
            'classification_report': {
                'Negative': {'precision': 0.46, 'recall': 0.88, 'f1-score': 0.60, 'support': 181},
                'Neutral': {'precision': 0.40, 'recall': 0.11, 'f1-score': 0.17, 'support': 18},
                'Positive': {'precision': 0.90, 'recall': 0.55, 'f1-score': 0.68, 'support': 398}
            }
        },
        'bert': {
            'accuracy': 0.7522,
            'f1': 0.750,
            'precision': 0.4800,
            'recall': 0.5100,
            'time': 8280,
            'confusion_matrix': np.array([[101, 8, 27], 
                                          [2, 0, 11], 
                                          [29, 34, 236]]),
            'classification_report': {
                'Negative': {'precision': 0.60, 'recall': 0.74, 'f1-score': 0.66, 'support': 136},
                'Neutral': {'precision': 0.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 13},
                'Positive': {'precision': 0.84, 'recall': 0.79, 'f1-score': 0.82, 'support': 299}
            },
            'history': {
                'epochs': [1, 2, 3],
                'val_acc': [0.7606, 0.7718, 0.7696],
                'val_loss': [0.5647, 0.5644, 0.6415]
            }
        }
    }

def get_model_performance(model_type, params, X_train, X_test, y_train, y_test):
    """Train and evaluate a model with given parameters"""
    if model_type == "Logistic Regression":
        model = LogisticRegression(
            C=params['C'],
            max_iter=params['max_iter'],
            class_weight=params['class_weight'],
            solver=params['solver'],
            random_state=42
        )
    elif model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            class_weight=params['class_weight'],
            random_state=42
        )
    elif model_type == "SVC":
        model = SVC(
            C=params['C'],
            kernel=params['kernel'],
            gamma=params['gamma'],
            class_weight=params['class_weight'],
            probability=True,
            random_state=42
        )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # For multi-class ROC AUC (one-vs-rest)
    if y_pred_proba is not None:
        try:
            # Encode labels for ROC AUC
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_test_encoded = le.fit_transform(y_test)
            roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = None
    else:
        roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def create_confusion_matrix_figure(cm, class_names, title):
    """Create a styled confusion matrix figure"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale=['#F5F7F5', '#8FA998', '#4A7C59', '#2C5530'],
        showscale=True,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        width=600
    )
    
    return fig

def create_wordcloud(text_data, title, color_scheme):
    """Create word cloud with specific color scheme"""
    text = ' '.join(text_data.dropna().astype(str))
    
    if color_scheme == 'blue':
        colormap = 'Blues'
    elif color_scheme == 'red':
        colormap = 'Reds'
    else:  # neutral
        colormap = 'Greys'
    
    wordcloud = WordCloud(
        width=400,
        height=300,
        background_color='white',
        colormap=colormap,
        max_words=100,
        contour_width=1,
        contour_color='#333'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, color='#2C5530', pad=20)
    return fig


# Sidebar navigation
st.sidebar.title("🤖 AI Sentiment Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📖 Guide", "📊 Data Explorer", "🔍 EDA", 
     "⚙️ Feature Engineering", "🤖 ML Models", "🧠 Deep Learning", "⚖️ Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project by:** Archisha Bhatt  
**Course:** CMSE 830  
**Institution:** Michigan State University
""")

# Load data
df = load_data()
cached_results = load_model_results()  # Renamed to avoid confusion

# Main content
if page == "🏠 Overview":
    st.title("🤖 AI Sentiment Analysis Dashboard")
    st.markdown("### Analyzing Public Discourse on Artificial Intelligence")
    
    st.markdown("""
    <div class='card'>
    <h3>Project Goal</h3>
    <p style='font-size: 16px; line-height: 1.6;'>
    This comprehensive analysis explores public sentiment towards Artificial Intelligence through 
    user comments from Reddit and YouTube. Using both traditional machine learning and deep learning 
    approaches, we uncover patterns, trends, and insights in AI-related discussions. We also use these
    models to help predict sentiment in new comments about AI.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='perf-card'>
                <h3 style='margin: 0; color: #4A7C59;'>{len(df):,}</h3>
                <p style='margin: 0; color: #6B8E7F;'>Total Comments</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sources = df['source'].nunique() if 'source' in df.columns else 2
            st.markdown(f"""
            <div class='perf-card'>
                <h3 style='margin: 0; color: #4A7C59;'>{sources}</h3>
                <p style='margin: 0; color: #6B8E7F;'>Data Sources</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='perf-card'>
                <h3 style='margin: 0; color: #4A7C59;'>4</h3>
                <p style='margin: 0; color: #6B8E7F;'>ML Models</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            bert_acc = cached_results['bert']['accuracy'] if 'bert' in cached_results else 0
            st.markdown(f"""
            <div class='perf-card'>
                <h3 style='margin: 0; color: #4A7C59;'>{bert_acc:.1%}</h3>
                <p style='margin: 0; color: #6B8E7F;'>Best Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
    <h3>Project Overview</h3>
    <p style='font-size: 16px; line-height: 1.6;'>
    The project followed a systematic data science workflow: <br>
    1. <b>Data Collection:</b> Scraped comments from Reddit and YouTube using APIs. <br>
    2. <b>Data Preprocessing:</b> Cleaned and prepared text data for analysis. <br>
    3. <b>Exploratory Data Analysis (EDA):</b> Visualized sentiment distributions and text characteristics. <br>
    4. <b>Feature Engineering:</b> Created text-based, statistical, and AI-specific features. <br>
    5. <b>Modeling:</b> Created tf-idf vectors and trained and evaluated Logistic Regression, Random Forest, SVC, and BERT models. <br>
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick insights
    if df is not None and 'sentiment' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='',
                color_discrete_map={
                    'Positive': '#3498db',
                    'Negative': '#e74c3c',
                    'Neutral': '#95a5a6'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Model Performance")
            
            model_names = []
            accuracies = []
            
            for model_name in ['logistic_regression', 'random_forest', 'svc', 'bert']:
                if model_name in cached_results:
                    model_names.append(model_name.replace('_', ' ').title())
                    accuracies.append(cached_results[model_name]['accuracy'])
            
            fig = px.bar(
                x=model_names,
                y=accuracies,
                title='',
                labels={'x': 'Model', 'y': 'Accuracy'},
                color=model_names,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=300, yaxis_tickformat='.0%', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

elif page == "📖 Guide":
    st.title("📖 Project Guide")
    
    st.markdown("""
    <div class='card'>
    <h3>How to Navigate This Dashboard</h3>
    <p>Use the sidebar to explore different sections of the analysis:</p>
    </div>
    """, unsafe_allow_html=True)
    
    guide_sections = {
        "📊 Data Explorer": {
            "description": "Explore the raw and processed data",
            "features": [
                "View sample comments",
                "Examine data statistics",
                "Check data quality metrics"
            ]
        },
        "🔍 EDA": {
            "description": "Exploratory Data Analysis visualizations",
            "features": [
                "Sentiment distribution analysis",
                "Word clouds by sentiment",
                "Temporal trend analysis",
                "Text characteristics"
            ]
        },
        "⚙️ Feature Engineering": {
            "description": "Understanding engineered features",
            "features": [
                "Text-based features",
                "Statistical features",
                "AI-specific features",
                "Feature calculation examples"
            ]
        },
        "🤖 ML Models": {
            "description": "Traditional machine learning models",
            "features": [
                "Logistic Regression",
                "Random Forest",
                "Support Vector Classifier",
                "Performance metrics and confusion matrices"
            ]
        },
        "🧠 Deep Learning": {
            "description": "BERT transformer model",
            "features": [
                "BERT architecture overview",
                "Fine-tuning process",
                "Performance evaluation",
                "Training history"
            ]
        },
        "⚖️ Comparison": {
            "description": "Model comparison and evaluation",
            "features": [
                "Side-by-side model comparison",
                "Performance metrics",
                "Confusion matrix comparison",
                "Recommendations"
            ]
        }
    }
    
    for section, content in guide_sections.items():
        with st.expander(f"### {section}", expanded=True):
            st.markdown(f"**{content['description']}**")
            st.markdown("**Features:**")
            for feature in content['features']:
                st.markdown(f"- {feature}")

elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    
    if df is None:
        st.error("Unable to load data.")
    else:
        tabs = st.tabs(["📋 Overview", "🔍 Sample Data", "📈 Statistics", "✅ Data Quality"])
        
        with tabs[0]:
            st.markdown("### Dataset Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Features", f"{len(df.columns)}")
            with col3:
                st.metric("Time Range", f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else "N/A")
            
            st.markdown("#### Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str).values,
                'Non-Null Count': df.count().values
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tabs[1]:
            st.markdown("### Sample Data")
            
            sample_size = st.slider("Number of samples", 5, 50, 10)
            
            # Select columns to display
            default_cols = ['text', 'sentiment', 'source', 'timestamp'] if all(col in df.columns for col in ['text', 'sentiment', 'source', 'timestamp']) else df.columns.tolist()[:5]
            selected_cols = st.multiselect(
                "Select columns to display",
                df.columns.tolist(),
                default=default_cols
            )
            
            if selected_cols:
                st.dataframe(df[selected_cols].head(sample_size), use_container_width=True, height=400)
        
        with tabs[2]:
            st.markdown("### Statistical Summary")
            
            if 'sentiment' in df.columns:
                st.markdown("#### Sentiment Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_counts = df['sentiment'].value_counts()
                    st.dataframe(sentiment_counts.to_frame('Count'), use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        labels={'x': 'Sentiment', 'y': 'Count'},
                        title='',
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'Positive': '#3498db',
                            'Negative': '#e74c3c',
                            'Neutral': '#95a5a6'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Numerical features summary
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                st.markdown("#### Numerical Features Summary")
                st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        
        with tabs[3]:
            st.markdown("### Data Quality Assessment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Check for missing values
                missing_total = df.isnull().sum().sum()
                st.metric("Missing Values", missing_total)
                if missing_total == 0:
                    st.success("✅ No missing values")
            
            with col2:
                # Check for duplicates
                duplicates = df.duplicated().sum()
                st.metric("Duplicate Rows", duplicates)
                if duplicates == 0:
                    st.success("✅ No duplicates")
            
            with col3:
                # Check data types
                text_cols = df.select_dtypes(include=['object']).columns
                st.metric("Text Columns", len(text_cols))
            
            st.markdown("#### Data Type Distribution")
            dtype_counts = df.dtypes.value_counts()
            fig = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index.astype(str),
                title='Data Types in Dataset'
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    
    if df is None:
        st.error("Unable to load data.")
    else:
        tabs = st.tabs(["📊 Distributions", "☁️ Word Clouds", "📈 Trends", "📏 Text Analysis"])
        
        with tabs[0]:
            st.markdown("### Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sentiment' in df.columns and 'source' in df.columns:
                    # Sentiment by source
                    source_sentiment = pd.crosstab(df['source'], df['sentiment'], normalize='index') * 100
                    
                    fig = px.bar(
                        source_sentiment,
                        title='Sentiment Distribution by Source (%)',
                        barmode='group',
                        color_discrete_map={
                            'Positive': '#3498db',
                            'Negative': '#e74c3c',
                            'Neutral': '#95a5a6'
                        }
                    )
                    fig.update_layout(xaxis_title='Source', yaxis_title='Percentage', height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'text_length' in df.columns:
                    # Text length distribution by sentiment
                    fig = px.box(
                        df,
                        x='sentiment',
                        y='text_length',
                        title='Text Length by Sentiment',
                        color='sentiment',
                        color_discrete_map={
                            'Positive': '#3498db',
                            'Negative': '#e74c3c',
                            'Neutral': '#95a5a6'
                        }
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.markdown("### Word Clouds by Sentiment")
            
            if 'sentiment' in df.columns and 'cleaned_text' in df.columns:
                max_words = st.slider("Maximum words in word clouds", 50, 300, 100)
                
                sentiments = df['sentiment'].unique()
                cols = st.columns(len(sentiments))
                
                color_schemes = {
                    'Positive': 'blue',
                    'Negative': 'red',
                    'Neutral': 'grey'
                }
                
                for idx, sentiment in enumerate(sentiments):
                    with cols[idx]:
                        sentiment_text = df[df['sentiment'] == sentiment]['cleaned_text']
                        
                        if len(sentiment_text) > 0:
                            fig = create_wordcloud(
                                sentiment_text, 
                                f"{sentiment} Sentiment", 
                                color_schemes.get(sentiment, 'grey')
                            )
                            st.pyplot(fig)
                            plt.close()
        
        with tabs[2]:
            st.markdown("### Temporal Trends")
            
            if 'year' in df.columns and 'timestamp' in df.columns:
                # Yearly analysis
                st.markdown("#### Yearly Comment Volume")
                
                yearly_counts = df.groupby('year').size().reset_index(name='count')
                fig = px.bar(
                    yearly_counts,
                    x='year',
                    y='count',
                    title='Comments by Year',
                    color_discrete_sequence=['#4A7C59']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                
                # Sentiment over time
                if 'sentiment' in df.columns:
                    st.markdown("#### Sentiment Trends Over Time")
                    
                    # Yearly sentiment
                    yearly_sentiment = df.groupby(['year', 'sentiment']).size().reset_index(name='count')
                    
                    fig = px.line(
                        yearly_sentiment,
                        x='year',
                        y='count',
                        color='sentiment',
                        title='Sentiment Trends by Year',
                        color_discrete_map={
                            'Positive': '#3498db',
                            'Negative': '#e74c3c',
                            'Neutral': '#95a5a6'
                        },
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            st.markdown("### Text Characteristics")
            
            if 'word_count' in df.columns and 'avg_word_length' in df.columns:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Words", f"{df['word_count'].mean():.1f}")
                
                with col2:
                    st.metric("Average Word Length", f"{df['avg_word_length'].mean():.2f}")
                
                with col3:
                    st.metric("Max Words", f"{df['word_count'].max()}")
                
                # Correlation matrix
                st.markdown("#### Feature Correlations")
                
                numerical_features = ['text_length', 'word_count', 'avg_word_length', 
                                    'ai_keyword_count', 'exclamation_count', 'question_count']
                numerical_features = [f for f in numerical_features if f in df.columns]
                
                if len(numerical_features) > 0:
                    corr_matrix = df[numerical_features].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='YlGn',
                        title='Feature Correlation Matrix'
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")
    
    st.markdown("""
    <div class='card'>
    <h3>Engineered Features for Sentiment Analysis</h3>
    <p>The following features were created from the raw text data to enhance model performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.error("Unable to load data.")
        st.stop()
    
    tabs = st.tabs(["📝 Text Features", "🤖 AI-Specific Features", "🎯 Statistical Features", "📊 Sample Calculations"])
    
    with tabs[0]:
        st.markdown("### Text-Based Features")
        
        text_features = [
            ("text_length", "Length of text in characters", "len(text)"),
            ("word_count", "Number of words in text", "len(text.split())"),
            ("avg_word_length", "Average length of words", "sum(len(word) for word in text.split()) / word_count"),
            ("sentence_count", "Number of sentences (approximate)", "text.count('.') + text.count('!') + text.count('?')")
        ]
        
        for feature, description, formula in text_features:
            if feature in df.columns:
                with st.expander(f"**{feature}**"):
                    st.markdown(f"**Description:** {description}")
                    st.markdown(f"**Formula:** `{formula}`")
                    st.markdown(f"**Range:** {df[feature].min():.1f} to {df[feature].max():.1f}")
                    st.markdown(f"**Mean:** {df[feature].mean():.2f}")
                    
                    # Show distribution
                    fig = px.histogram(
                        df, 
                        x=feature, 
                        nbins=50,
                        title=f'{feature} Distribution',
                        color_discrete_sequence=['#4A7C59']
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### AI-Specific Features")
        
        st.markdown("""
        These features capture domain-specific aspects of AI-related discussions:
        """)
        
        if 'ai_keyword_count' in df.columns:
            st.markdown("#### AI Keyword Count")
            st.markdown("Counts occurrences of AI-related terms in the text.")
            
            ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 
                          'neural network', 'algorithm', 'chatgpt', 'gpt', 'llm', 'transformer']
            
            st.markdown("**Keywords tracked:** " + ", ".join(ai_keywords))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average AI keywords per comment", f"{df['ai_keyword_count'].mean():.2f}")
            
            with col2:
                st.metric("Comments with AI keywords", 
                         f"{(df['ai_keyword_count'] > 0).sum() / len(df) * 100:.1f}%")
            
            # Show distribution
            fig = px.histogram(
                df, 
                x='ai_keyword_count', 
                nbins=20,
                title='AI Keyword Count Distribution',
                color_discrete_sequence=['#3498db']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### Statistical Features")
        
        stat_features = [
            ("exclamation_count", "Number of exclamation marks", "Indicates emotional intensity"),
            ("question_count", "Number of question marks", "Indicates inquiry or uncertainty"),
            ("has_question", "Whether text contains questions", "Binary indicator"),
            ("has_exclamation", "Whether text contains exclamations", "Binary indicator")
        ]
        
        for feature, description, purpose in stat_features:
            if feature in df.columns:
                with st.expander(f"**{feature}**"):
                    st.markdown(f"**Description:** {description}")
                    st.markdown(f"**Purpose:** {purpose}")
                    
                    if feature in ['has_question', 'has_exclamation']:
                        # For binary features
                        value_counts = df[feature].value_counts()
                        fig = px.pie(
                            values=value_counts.values,
                            names=value_counts.index.astype(str),
                            title=f'{feature} Distribution'
                        )
                    else:
                        # For count features
                        fig = px.histogram(
                            df, 
                            x=feature, 
                            nbins=20,
                            title=f'{feature} Distribution',
                            color_discrete_sequence=['#e74c3c']
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("### Feature Calculation Examples")
        
        st.markdown("""
        Here's how features are calculated for sample comments:
        """)
        
        # Show sample calculations
        sample_idx = st.selectbox("Select sample comment", range(min(5, len(df))))
        
        sample_data = df.iloc[sample_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Text:**")
            st.text_area("", sample_data.get('text', 'N/A'), height=150, disabled=True)
        
        with col2:
            st.markdown("**Calculated Features:**")
            
            feature_values = []
            for feature in ['text_length', 'word_count', 'avg_word_length', 
                          'ai_keyword_count', 'exclamation_count', 'question_count']:
                if feature in df.columns:
                    value = sample_data[feature]
                    feature_values.append(f"{feature}: {value}")
            
            st.text_area("", "\n".join(feature_values), height=150, disabled=True)
        
        # Show formula application
        st.markdown("#### How features are calculated:")
        
        if 'text' in sample_data:
            text = str(sample_data['text'])
            
            calculations = [
                f"text_length = len('{text[:50]}...') = {len(text)}",
                f"word_count = len('{text[:50]}...'.split()) = {len(text.split())}",
                f"avg_word_length = sum(len(word) for word in text.split()) / word_count = {np.mean([len(w) for w in text.split()]):.2f}" if len(text.split()) > 0 else "No words to calculate average"
            ]
            
            for calc in calculations:
                st.code(calc, language='python')

elif page == "🤖 ML Models":
    st.title("🤖 Interactive Machine Learning Models")
    st.markdown("### Real-time parameter tuning and model training")
    
    st.info("""
    ### 🎯 How to use this interactive ML lab:
    
    1. **Configure your data** in the Data Preparation section below
    2. **Select a model type** (Logistic Regression, Random Forest, or SVC)
    3. **Adjust the parameters** using the sliders and dropdowns
    4. **Click 'Train Model with Current Parameters'** to train and evaluate
    5. **Explore the results** including metrics, confusion matrix, and predictions
    
    ⚡ **Real-time updates**: Every time you change parameters and click train, you'll get fresh results!
    """)
    
    if df is None:
        st.error("Unable to load data. Please ensure 'data/final_df.csv' exists.")
        st.stop()
    
    # Data preparation section
    with st.expander("📊 Data Preparation & Setup", expanded=True):
        st.markdown("#### Configure your training data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
        
        with col2:
            vectorizer_type = st.selectbox(
                "Text Vectorization",
                ["TF-IDF", "Count Vectorizer"]
            )
            
            use_tfidf = st.checkbox("Use TF-IDF features", value=True)
        
        with col3:
            include_numeric = st.checkbox("Include numeric features", value=True)
            random_state = st.number_input("Random State", 1, 100, 42)
        
        # Prepare features
        if 'cleaned_text' not in df.columns:
            st.error("No cleaned text column found in data")
            st.stop()
        
        # Create text features
        if use_tfidf:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            X_text = vectorizer.fit_transform(df['cleaned_text'])
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            X_text = vectorizer.fit_transform(df['cleaned_text'])
        
        # Add numeric features if selected
        if include_numeric:
            # Create some numeric features
            df['text_length'] = df['cleaned_text'].str.len()
            df['word_count'] = df['cleaned_text'].str.split().str.len()
            
            numeric_features = ['text_length', 'word_count']
            X_numeric = df[numeric_features].values
            
            # Scale numeric features
            scaler = StandardScaler()
            X_numeric_scaled = scaler.fit_transform(X_numeric)
            
            # Combine features
            X = hstack([X_text, X_numeric_scaled])
        else:
            X = X_text
        
        # Prepare target
        if 'sentiment' not in df.columns:
            st.error("No sentiment column found in data")
            st.stop()
        
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        st.success(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        st.info(f"**Features:** {X.shape[1]} | **Classes:** {y.nunique()} ({', '.join(y.unique())})")
    
    # Model selection
    st.markdown("---")
    st.markdown("### 🎯 Select and Configure Model")
    
    model_type = st.selectbox(
        "Choose Model Type",
        ["Logistic Regression", "Random Forest", "SVC"],
        key="model_selector"
    )
    
    # Model-specific parameter controls
    if model_type == "Logistic Regression":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Logistic Regression Parameters")
            C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
            max_iter = st.slider("Max Iterations", 100, 5000, 1000, 100)
            solver = st.selectbox("Solver", ["liblinear", "lbfgs", "sag", "saga"])
        
        with col2:
            class_weight = st.selectbox("Class Weight", ["balanced", None])
            penalty = st.selectbox("Penalty", ["l2", "l1"])
            multi_class = st.selectbox("Multi-class", ["ovr", "multinomial"])
        
        params = {
            'C': C,
            'max_iter': max_iter,
            'solver': solver,
            'class_weight': class_weight,
            'penalty': penalty,
            'multi_class': multi_class
        }
    
    elif model_type == "Random Forest":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Random Forest Parameters")
            n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
            max_depth = st.selectbox("Max Depth", [None, 5, 10, 20, 30, 50])
        
        with col2:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
            class_weight = st.selectbox("Class Weight", ["balanced", "balanced_subsample", None])
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight
        }
    
    elif model_type == "SVC":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### SVC Parameters")
            C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        
        with col2:
            gamma = st.selectbox("Gamma", ["scale", "auto", 0.1, 0.01, 0.001])
            class_weight = st.selectbox("Class Weight", ["balanced", None])
            degree = st.slider("Degree (for poly kernel)", 2, 5, 3) if kernel == "poly" else 3
        
        params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'class_weight': class_weight,
            'degree': degree
        }
    
    # Train and evaluate button
    st.markdown("---")
    if st.button("🚀 Train Model with Current Parameters", type="primary", use_container_width=True):
        with st.spinner(f"Training {model_type}..."):
            # Train and evaluate model
            current_results = get_model_performance(
                model_type, 
                params, 
                X_train, X_test, y_train, y_test
            )
            
            # Store results in session state
            st.session_state['current_results'] = current_results
            st.session_state['current_model_type'] = model_type
            st.session_state['current_params'] = params
    
    # Display results if available
    if 'current_results' in st.session_state:
        current_results = st.session_state['current_results']
        model_type = st.session_state['current_model_type']
        params = st.session_state['current_params']
        
        st.markdown("---")
        st.markdown(f"### 📊 {model_type} Results")
        
        # Performance metrics
        st.markdown("#### Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="perf-card">
                <h3 style="color: #4A7C59; margin: 0;">{current_results['accuracy']:.2%}</h3>
                <p style="margin: 0; color: #6B8E7F;">Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="perf-card">
                <h3 style="color: #4A7C59; margin: 0;">{current_results['f1']:.3f}</h3>
                <p style="margin: 0; color: #6B8E7F;">F1 Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="perf-card">
                <h3 style="color: #4A7C59; margin: 0;">{current_results['precision']:.3f}</h3>
                <p style="margin: 0; color: #6B8E7F;">Precision</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="perf-card">
                <h3 style="color: #4A7C59; margin: 0;">{current_results['recall']:.3f}</h3>
                <p style="margin: 0; color: #6B8E7F;">Recall</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            if current_results['roc_auc']:
                st.markdown(f"""
                <div class="perf-card">
                    <h3 style="color: #4A7C59; margin: 0;">{current_results['roc_auc']:.3f}</h3>
                    <p style="margin: 0; color: #6B8E7F;">ROC AUC</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="perf-card">
                    <h3 style="color: #4A7C59; margin: 0;">N/A</h3>
                    <p style="margin: 0; color: #6B8E7F;">ROC AUC</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        class_names = sorted(y.unique())
        cm_fig = create_confusion_matrix_figure(
            current_results['confusion_matrix'],
            class_names,
            f'{model_type} Confusion Matrix'
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        
        # Classification Report
        st.markdown("#### Detailed Classification Report")
        
        # Convert classification report to DataFrame
        report_df = pd.DataFrame(current_results['classification_report']).transpose()
        
        # Style the DataFrame
        def color_metric(val):
            if isinstance(val, (int, float)):
                if val >= 0.8:
                    return 'background-color: #d4edda; color: #155724;'
                elif val >= 0.6:
                    return 'background-color: #fff3cd; color: #856404;'
                else:
                    return 'background-color: #f8d7da; color: #721c24;'
            return ''
        
        styled_report = report_df.style.applymap(color_metric, subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']])
        st.dataframe(styled_report, use_container_width=True)
        
        # Model Insights
        st.markdown("#### Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Current Parameters")
            param_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
            st.dataframe(param_df, use_container_width=True)
        
        with col2:
            st.markdown("##### Class Distribution")
            class_dist = pd.DataFrame({
                'Class': class_names,
                'Train Count': [sum(y_train == cls) for cls in class_names],
                'Test Count': [sum(y_test == cls) for cls in class_names]
            })
            class_dist['Train %'] = class_dist['Train Count'] / len(y_train) * 100
            class_dist['Test %'] = class_dist['Test Count'] / len(y_test) * 100
            
            fig = px.bar(
                class_dist,
                x='Class',
                y=['Train Count', 'Test Count'],
                barmode='group',
                title='Class Distribution in Train/Test Sets',
                color_discrete_map={
                    'Train Count': '#4A7C59',
                    'Test Count': '#3498db'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample predictions
        st.markdown("#### Sample Predictions")

        # Get a sample of test predictions
        n_samples = min(10, len(y_test))

        # Get random indices as Python integers
        sample_indices = np.random.choice(len(y_test), n_samples, replace=False).tolist()        

        sample_data = []
        for idx in sample_indices:
            # Convert idx to Python int
            idx = int(idx)
    
            # Get actual value
            if isinstance(y_test, pd.Series):
                actual = y_test.iloc[idx]
            else:
                actual = y_test[idx]
    
            # Get predicted value
            predicted = current_results['predictions'][idx]
    
            # Get confidence if available
            if current_results['probabilities'] is not None:
                probs = current_results['probabilities'][idx]
                confidence = max(probs)
                predicted_class = class_names[np.argmax(probs)]
            else:
                confidence = None
                predicted_class = predicted
    
            # Get the corresponding text from original dataframe
            text = "N/A"
            if isinstance(y_test, pd.Series):
                # Get the original index from y_test
                original_index = y_test.index[idx]
                if original_index in df.index:
                    text = str(df.loc[original_index, 'cleaned_text'])
    
            # Truncate text if too long
            if len(text) > 100:
                display_text = text[:100] + '...'
            else:
                display_text = text
    
            sample_data.append({
                'Text': display_text,
                'Actual': actual,
                'Predicted': predicted_class,
                'Confidence': f"{confidence:.3f}" if confidence is not None else "N/A",
                'Correct': '✅' if actual == predicted_class else '❌'
            })

        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        # Feature importance (for interpretable models)
        if model_type == "Logistic Regression" and hasattr(current_results['model'], 'coef_'):
            st.markdown("#### Feature Importance")
            
            try:
                # Get feature names
                feature_names = []
                if use_tfidf:
                    feature_names.extend(vectorizer.get_feature_names_out())
                if include_numeric:
                    feature_names.extend(['text_length', 'word_count'])
                
                # Get coefficients (for multi-class, we have coefficients per class)
                coefficients = current_results['model'].coef_
                
                # Create a DataFrame for each class
                for i, class_name in enumerate(class_names):
                    if i < coefficients.shape[0]:
                        class_coef = coefficients[i]
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Coefficient': class_coef
                        })
                        
                        # Show top 10 positive and negative features
                        top_positive = importance_df.nlargest(10, 'Coefficient')
                        top_negative = importance_df.nsmallest(10, 'Coefficient')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Top Positive Features - {class_name}**")
                            fig = px.bar(
                                top_positive,
                                x='Coefficient',
                                y='Feature',
                                orientation='h',
                                color='Coefficient',
                                color_continuous_scale='Greens'
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"**Top Negative Features - {class_name}**")
                            fig = px.bar(
                                top_negative,
                                x='Coefficient',
                                y='Feature',
                                orientation='h',
                                color='Coefficient',
                                color_continuous_scale='Reds_r'
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not extract feature importance: {str(e)}")
        
        elif model_type == "Random Forest" and hasattr(current_results['model'], 'feature_importances_'):
            st.markdown("#### Feature Importance")
            
            try:
                # Get feature names
                feature_names = []
                if use_tfidf:
                    feature_names.extend(vectorizer.get_feature_names_out())
                if include_numeric:
                    feature_names.extend(['text_length', 'word_count'])
                
                # Get feature importances
                importances = current_results['model'].feature_importances_
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(20)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Blues',
                    title='Top 20 Feature Importances'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not extract feature importance: {str(e)}")
        
        # Comparison with baseline
        st.markdown("#### Performance Comparison")
        
        # Train a simple baseline model for comparison
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        
        comparison_data = pd.DataFrame({
            'Model': ['Baseline (Most Frequent)', model_type],
            'Accuracy': [baseline_accuracy, current_results['accuracy']],
            'F1 Score': [f1_score(y_test, baseline_pred, average='weighted'), current_results['f1']]
        })
        
        fig = px.bar(
            comparison_data,
            x='Model',
            y=['Accuracy', 'F1 Score'],
            barmode='group',
            title='Model vs Baseline Comparison',
            color_discrete_sequence=['#95a5a6', '#4A7C59']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.markdown("#### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Predictions", use_container_width=True):
                predictions_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': current_results['predictions']
                })
                
                if current_results['probabilities'] is not None:
                    prob_df = pd.DataFrame(
                        current_results['probabilities'],
                        columns=[f"prob_{cls}" for cls in class_names]
                    )
                    predictions_df = pd.concat([predictions_df, prob_df], axis=1)
                
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{model_type.lower().replace(' ', '_')}_predictions.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("💾 Save Model Parameters", use_container_width=True):
                import json
                params_json = json.dumps(params, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=params_json,
                    file_name=f"{model_type.lower().replace(' ', '_')}_params.json",
                    mime="application/json"
                )

elif page == "🧠 Deep Learning":
    st.title("🧠 BERT Deep Learning Model")
    
    st.markdown("""
    <div class='card'>
    <h3>BERT (Bidirectional Encoder Representations from Transformers)</h3>
    <p>Fine-tuned bert-base-uncased model for sentiment analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'bert' not in cached_results:
        st.error("BERT results not available.")
        st.stop()
    
    bert = cached_results['bert']
    
    tabs = st.tabs(["📊 Performance", "🎯 Confusion Matrix", "📈 Training History", "🔧 Model Details"])
    
    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{bert['accuracy']:.2%}")
        with col2:
            st.metric("F1 Score", f"{bert['f1']:.3f}")
        with col3:
            st.metric("Precision", f"{bert['precision']:.3f}")
        with col4:
            st.metric("Training Time", f"{bert['time']/3600:.1f} hours")
        
        st.markdown("#### Classification Report")
        report_data = []
        for class_name, metrics in bert['classification_report'].items():
            report_data.append({
                'Class': class_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            })
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### Confusion Matrix")
        cm_fig = create_confusion_matrix_figure(
            bert['confusion_matrix'],
            ['Negative', 'Neutral', 'Positive'],
            'BERT Confusion Matrix'
        )
        st.plotly_chart(cm_fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### Training History")
        
        if 'history' in bert:
            history = bert['history']
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Validation Accuracy', 'Validation Loss'))
            
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['val_acc'], 
                          mode='lines+markers', name='Val Acc', 
                          line=dict(color='#4A7C59', width=3)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['val_loss'], 
                          mode='lines+markers', name='Val Loss', 
                          line=dict(color='#e74c3c', width=3)),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("### Model Architecture")
        
        st.markdown("""
        **BERT-base-uncased Configuration:**
        - **Layers:** 12
        - **Hidden Size:** 768
        - **Attention Heads:** 12
        - **Total Parameters:** 110M
        - **Training Data:** Wikipedia + BookCorpus
        
        **Fine-tuning Parameters:**
        - **Learning Rate:** 2e-5
        - **Batch Size:** 16
        - **Epochs:** 3
        - **Max Sequence Length:** 128 tokens
        - **Optimizer:** AdamW
        """)
        
        st.markdown("#### Model Strengths")
        st.success("""
        - Contextual understanding of language
        - Handles negation and complex sentence structures
        - Pre-trained on massive text corpus
        - State-of-the-art for NLP tasks
        """)
        
        st.markdown("#### Limitations")
        st.warning("""
        - Computationally expensive to train
        - Requires GPU for reasonable training time
        - Large model size (440MB)
        - Struggles with very small neutral class
        """)
        
elif page == "⚖️ Comparison":
    st.title("⚖️ Model Comparison and Evaluation")
    
    st.markdown("""
    <div class='card'>
    <h3>Comprehensive Model Comparison</h3>
    <p>Side-by-side comparison of all trained models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not cached_results:
        st.error("No model results available.")
        st.stop()
    
    # Create comparison data
    comparison_data = []
    for model_name, model_info in cached_results.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': model_info['accuracy'],
            'F1 Score': model_info['f1'],
            'Precision': model_info['precision'],
            'Recall': model_info['recall'],
            'Training Time (s)': model_info['time'] if isinstance(model_info['time'], (int, float)) else model_info['time'],
            'Type': 'Deep Learning' if model_name == 'bert' else 'Traditional ML'
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.markdown("### 📊 Performance Comparison Table")
    
    # Format the dataframe
    styled_df = comp_df.style.format({
        'Accuracy': '{:.2%}',
        'F1 Score': '{:.3f}',
        'Precision': '{:.3f}',
        'Recall': '{:.3f}'
    }).background_gradient(subset=['Accuracy', 'F1 Score'], cmap='YlGn')
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Visualization
    st.markdown("### 📈 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = px.bar(
            comp_df,
            x='Model',
            y='Accuracy',
            title='Accuracy Comparison',
            color='Type',
            color_discrete_map={
                'Traditional ML': '#4A7C59',
                'Deep Learning': '#3498db'
            },
            text=comp_df['Accuracy'].apply(lambda x: f'{x:.1%}')
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # F1 Score comparison
        fig = px.bar(
            comp_df,
            x='Model',
            y='F1 Score',
            title='F1 Score Comparison',
            color='Type',
            color_discrete_map={
                'Traditional ML': '#4A7C59',
                'Deep Learning': '#3498db'
            },
            text=comp_df['F1 Score'].apply(lambda x: f'{x:.3f}')
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix Comparison
    st.markdown("### 🎯 Confusion Matrix Comparison")
    
    # Create subplots for confusion matrices
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{model.replace('_', ' ').title()} Confusion Matrix" 
                       for model in ['logistic_regression', 'random_forest', 'svc', 'bert']],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    class_names = ['Negative', 'Neutral', 'Positive']
    
    # Add confusion matrices to subplots
    row_col_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for idx, (model_name, position) in enumerate(zip(['logistic_regression', 'random_forest', 'svc', 'bert'], row_col_positions)):
        if model_name in cached_results:
            cm = cached_results[model_name]['confusion_matrix']
            row, col = position
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=class_names,
                    y=class_names,
                    colorscale='YlGn',
                    showscale=False,
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ),
                row=row, col=col
            )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.markdown("### 🔍 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Strengths by Model")
        
        strengths = {
            "Logistic Regression": [
                "Fastest traditional model",
                "Good interpretability",
                "Best handling of neutral class"
            ],
            "Random Forest": [
                "Robust to outliers",
                "Feature importance available",
                "Good overall balance"
            ],
            "Support Vector Classifier": [
                "Effective in high dimensions",
                "Good margin maximization",
                "Versatile kernel options"
            ],
            "BERT": [
                "Highest overall accuracy",
                "Contextual understanding",
                "State-of-the-art performance"
            ]
        }
        
        for model, model_strengths in strengths.items():
            with st.expander(f"**{model}**"):
                for strength in model_strengths:
                    st.markdown(f"✓ {strength}")
    
    with col2:
        st.markdown("#### Recommendations")
        
        scenarios = [
            ("Production Deployment", "Logistic Regression", "Fast inference, good accuracy, easy to maintain"),
            ("High Accuracy Requirement", "BERT", "Best performance but requires GPU resources"),
            ("Balanced Performance", "Random Forest", "Good trade-off between accuracy and speed"),
            ("Research/Experimentation", "BERT", "State-of-the-art, good for benchmarks"),
            ("Resource-Constrained", "Logistic Regression", "Minimal computational requirements")
        ]
        
        for scenario, recommendation, reason in scenarios:
            st.markdown(f"**{scenario}:**")
            st.markdown(f"*Recommend:* {recommendation}")
            st.markdown(f"*Reason:* {reason}")
            st.markdown("---")
    
    # Final summary
    st.markdown("### 🏆 Final Summary")
    
    best_model = comp_df.loc[comp_df['Accuracy'].idxmax()]
    
    st.success(f"""
    **Overall Best Model:** {best_model['Model']}
    
    **Key Findings:**
    1. **BERT** achieves the highest accuracy ({best_model['Accuracy']:.2%}) but at significant computational cost
    2. **Logistic Regression** offers the best balance of performance and efficiency among traditional models
    3. All models struggle with the **Neutral** class due to class imbalance
    4. **Traditional models** are 1000x faster to train than BERT
    5. For production systems requiring real-time inference, **Logistic Regression** is recommended
    """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B8E7F; padding: 1rem;'>
<p><b>AI Sentiment Analysis Project</b> | CMSE 830 - Foundations of Data Science</p>
<p>Created by Archisha Bhatt | Michigan State University | 2025</p>
</div>
""", unsafe_allow_html=True)