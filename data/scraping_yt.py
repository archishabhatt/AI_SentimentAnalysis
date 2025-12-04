import os
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
from datetime import datetime, timezone
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
def get_api_key():
    """Get API key from environment variable"""
    API_KEY = os.environ.get("YOUTUBE_API_KEY")
    if not API_KEY:
        raise ValueError(
            "Please set YOUTUBE_API_KEY environment variable.\n"
            "Linux/Mac: export YOUTUBE_API_KEY='your_key_here'\n"
            "Windows: set YOUTUBE_API_KEY=your_key_here"
        )
    return API_KEY

def create_youtube_service():
    """Create YouTube API service with proper configuration"""
    API_KEY = get_api_key()
    
    return build(
        "youtube", 
        "v3", 
        developerKey=API_KEY,
        cache_discovery=False
    )

# -----------------------------
# Content Filtering
# -----------------------------
# --- Keywords for filtering ---
ai_keywords = [
    "artificial intelligence", "AI", "GPT", "ChatGPT", "DeepSeek", "Claude", "Gemini",
    "LLM", "Large Language Model", "machine learning", "neural network", "transformer",
    "AGI", "Artificial General Intelligence", "generative AI"
]

opinion_keywords = [
    "think", "believe", "feel", "opinion", "perspective", "view", "standpoint",
    "should", "would", "could", "might", "argument", "debate", "discussion",
    "pro", "con", "advantage", "disadvantage", "benefit", "drawback"
]

societal_keywords = [
    "ethical", "ethics", "moral", "society", "societal", "impact", "consequence",
    "future", "humanity", "job", "employment", "economy", "privacy", "security",
    "bias", "fair", "transparent", "accountable", "responsible", "regulation"
]

# --- Date range ---
start_date = "2020-01-01T00:00:00Z"
end_date = "2025-12-31T23:59:59Z"

# --- Text filtering parameters ---
MIN_WORDS = 10
MAX_WORDS = 300
MIN_SENTENCES = 1  # Reduced from 2 to be less strict

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_word_count(text):
    """Get word count of text"""
    return len(str(text).split())

def contains_any_keyword(text, keywords):
    """Check if text contains any of the keywords (flexible matching)"""
    text_lower = text.lower()
    for keyword in keywords:
        # For multi-word keywords, check if all words appear (not necessarily as phrase)
        if ' ' in keyword:
            words = keyword.split()
            if all(word in text_lower for word in words):
                return True
        # For single word keywords
        elif keyword.lower() in text_lower:
            return True
    return False

def contains_ai_keywords(text):
    """Check if text contains AI-related keywords (more flexible)"""
    text_lower = text.lower()

    # Robust match for 'AI' as a word
    if re.search(r'\bai\b', text_lower):
        return True
    
    # Other acronyms (whole-word match)
    acronyms = ['gpt', 'llm', 'agi']
    for a in acronyms:
        if re.search(rf'\b{a}\b', text_lower):
            return True
    
    # Full phrases (in-order match)
    phrases = [
        'artificial intelligence', 'machine learning', 
        'neural network', 'deep learning', 
        'chatgpt', 'openai', 'claude', 'gemini'
    ]
    if any(phrase in text_lower for phrase in phrases):
        return True

    return False


def is_relevant_comment(text):
    """Check if comment is relevant for analysis"""
    # Clean text first
    cleaned_text = clean_text(text)
    
    # Check word count
    word_count = get_word_count(cleaned_text)
    if word_count < MIN_WORDS or word_count > MAX_WORDS:
        return False
    
    # Check if it has sentences
    sentences = re.split(r'[.!?]+', cleaned_text)
    valid_sentences = [s for s in sentences if len(s.strip()) > 5]
    if len(valid_sentences) < MIN_SENTENCES:
        return False
    
    # Must contain AI keywords (more flexible check)
    if not contains_ai_keywords(cleaned_text):
        return False
    
    # Must contain either opinion or societal keywords
    if not (contains_any_keyword(cleaned_text, opinion_keywords) or 
            contains_any_keyword(cleaned_text, societal_keywords)):
        return False
    
    # Must be somewhat subjective (not purely factual)
    try:
        if TextBlob(cleaned_text).sentiment.subjectivity < 0.15:  # Reduced threshold
            return False
    except:
        return False
    
    return True

# -----------------------------
# Video Search with Date Filtering
# -----------------------------
def search_videos(youtube, query, max_results=20, published_after=None, published_before=None):
    """Search for videos based on query with date filtering"""
    try:
        request_params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "order": "relevance",
            "videoCaption": "any",
            "videoDefinition": "any",
            "videoDuration": "any"
        }
        
        # Add date filters if provided
        if published_after:
            request_params["publishedAfter"] = published_after
        if published_before:
            request_params["publishedBefore"] = published_before
        
        results = youtube.search().list(**request_params).execute()
        
        video_ids = [item["id"]["videoId"] for item in results.get("items", [])]
        
        logger.info(f"Found {len(video_ids)} videos for query: '{query}'")
        
        # Return all videos, we'll filter later
        return video_ids
            
    except HttpError as e:
        logger.error(f"Error searching videos: {e}")
        return []

# -----------------------------
# Video Metadata (Batch)
# -----------------------------
def get_videos_metadata(youtube, video_ids):
    """Get metadata for multiple videos at once"""
    try:
        # YouTube API allows up to 50 IDs per request
        chunks = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]
        all_metadata = {}
        
        for chunk in chunks:
            response = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(chunk)
            ).execute()
            
            for item in response.get("items", []):
                video_id = item["id"]
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                
                # Parse published date
                published_at = snippet.get("publishedAt", "")
                try:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                except:
                    pub_date = None
                
                all_metadata[video_id] = {
                    "video_title": snippet.get("title", ""),
                    "video_description": snippet.get("description", ""),
                    "video_published_at": published_at,
                    "video_published_date": pub_date,
                    "video_channel": snippet.get("channelTitle", ""),
                    "video_view_count": int(stats.get("viewCount", 0)),
                    "video_like_count": int(stats.get("likeCount", 0)),
                    "video_comment_count": int(stats.get("commentCount", 0))
                }
        
        return all_metadata
        
    except HttpError as e:
        logger.error(f"Error getting video metadata: {e}")
        return {}

# -----------------------------
# Comment Extraction with Filtering
# -----------------------------
def get_video_comments(youtube, video_id, video_meta, max_comments=100):
    """Extract and filter comments from a video"""
    comments = []
    next_page_token = None
    attempts = 0
    max_attempts = 3
    
    try:
        # Check if comments are enabled and have reasonable count
        comment_count = video_meta.get("video_comment_count", 0)
        if comment_count < 5:  # Reduced threshold
            logger.info(f"Video {video_id} has only {comment_count} comments, skipping")
            return comments
        
        logger.info(f"Scraping comments from video: {video_meta.get('video_title', 'Unknown')[:50]}...")
        
        while len(comments) < max_comments and attempts < max_attempts:
            try:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=next_page_token,
                    textFormat="plainText",
                    order="relevance"  # Get most relevant comments first
                ).execute()
                
                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    raw_text = snippet.get("textDisplay", "").strip()
                    
                    # Clean text
                    cleaned_text = clean_text(raw_text)
                    
                    # Apply relevance filters
                    if cleaned_text and is_relevant_comment(cleaned_text):
                        
                        word_count = get_word_count(cleaned_text)
                        try:
                            sentiment = TextBlob(cleaned_text).sentiment
                        except:
                            sentiment = type('obj', (object,), {'polarity': 0.0, 'subjectivity': 0.5})()
                        
                        comments.append({
                            # --- TEXT & ANALYSIS ---
                            "text": cleaned_text,
                            "comment_length": word_count,
                            "sentiment_polarity": sentiment.polarity,
                            "sentiment_subjectivity": sentiment.subjectivity,
                            
                            # --- ENGAGEMENT ---
                            "likes": snippet.get("likeCount", 0),
                            "num_replies": item.get("snippet", {}).get("totalReplyCount", 0),
                            
                            # --- TIME ---
                            "created_utc": snippet.get("publishedAt"),
                            
                            # --- SOURCE IDENTIFIERS ---
                            "source": "youtube",
                            "source_id": f"youtube_comment_{snippet.get('id')}",
                            "video_id": video_id,
                            
                            # --- VIDEO METADATA ---
                            "video_title": video_meta.get("video_title", ""),
                            "video_channel": video_meta.get("video_channel", ""),
                            "video_view_count": video_meta.get("video_view_count", 0),
                            "video_like_count": video_meta.get("video_like_count", 0),
                            "video_comment_count": video_meta.get("video_comment_count", 0),
                            "video_published_at": video_meta.get("video_published_at", ""),
                            
                            # --- CATEGORY FLAGS ---
                            "contains_ai": contains_ai_keywords(cleaned_text),
                            "contains_opinion": contains_any_keyword(cleaned_text, opinion_keywords),
                            "contains_societal": contains_any_keyword(cleaned_text, societal_keywords),
                        })
                        
                        if len(comments) >= max_comments:
                            break
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
                # Rate limiting
                time.sleep(0.1)
                attempts = 0  # Reset attempts on success
                
            except HttpError as e:
                if e.resp.status == 403 and ("commentsDisabled" in str(e) or "disabled" in str(e)):
                    logger.warning(f"Comments disabled for video: {video_id}")
                    break
                elif e.resp.status == 403 and "quotaExceeded" in str(e):
                    logger.error("API quota exceeded!")
                    break
                else:
                    attempts += 1
                    logger.warning(f"API error (attempt {attempts}/{max_attempts}): {e}")
                    if attempts < max_attempts:
                        time.sleep(2)
                        continue
                    else:
                        break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                attempts += 1
                if attempts < max_attempts:
                    time.sleep(2)
                    continue
                else:
                    break
        
        logger.info(f"Collected {len(comments)} relevant comments from video {video_id}")
        return comments
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        return []

# -----------------------------
# Main Pipeline
# -----------------------------
def scrape_youtube_ai_data(queries=None, videos_per_query=3, comments_per_video=30):
    """Main function to scrape YouTube AI discussions"""
    
    if queries is None:
        queries = [
            "AI ethics",
            "artificial intelligence impact",
            "ChatGPT debate",
            "future of AI",
            "machine learning bias",
            "LLM risks"
        ]
    
    # Initialize YouTube service
    youtube = create_youtube_service()
    
    all_comments = []
    total_videos_processed = 0
    
    for query_index, query in enumerate(queries):
        logger.info(f"\n{'='*60}")
        logger.info(f"Query {query_index + 1}/{len(queries)}: '{query}'")
        logger.info(f"{'='*60}")
        
        # Search for videos with date filtering
        video_ids = search_videos(
            youtube, 
            query, 
            max_results=10,  # Reduced from 15
            published_after=start_date,
            published_before=end_date
        )
        
        if not video_ids:
            logger.warning(f"No videos found for query: {query}")
            continue
        
        # Get metadata for all videos
        videos_metadata = get_videos_metadata(youtube, video_ids)
        
        # Process videos
        videos_processed = 0
        for i, video_id in enumerate(video_ids):
            if videos_processed >= videos_per_query:
                break
            
            if video_id not in videos_metadata:
                continue
                
            video_meta = videos_metadata[video_id]
            
            # Simple filtering - just check if video has enough comments
            if video_meta.get("video_comment_count", 0) >= 10:
                
                logger.info(f"\nProcessing video {i+1}/{len(video_ids)}: {video_meta['video_title'][:60]}...")
                logger.info(f"Channel: {video_meta['video_channel']}")
                logger.info(f"Comments: {video_meta['video_comment_count']}")
                
                # Scrape comments
                video_comments = get_video_comments(
                    youtube, 
                    video_id, 
                    video_meta,
                    max_comments=comments_per_video
                )
                
                if video_comments:
                    all_comments.extend(video_comments)
                    videos_processed += 1
                    total_videos_processed += 1
                    
                    logger.info(f"Added {len(video_comments)} comments (Total: {len(all_comments)})")
                else:
                    logger.info(f"No relevant comments found in this video")
                
                # Rate limiting between videos
                time.sleep(0.5)
        
        # Rate limiting between queries
        if query_index < len(queries) - 1:
            time.sleep(1)
    
    return all_comments

# -----------------------------
# Test Functions
# -----------------------------
def test_youtube_api():
    """Test if YouTube API is working"""
    try:
        youtube = create_youtube_service()
        
        # Simple test search
        request = youtube.search().list(
            part="snippet",
            q="artificial intelligence",
            type="video",
            maxResults=5
        )
        response = request.execute()
        
        print(f"\nAPI Test Results:")
        print(f"API key valid: YES")
        print(f"Videos found: {len(response.get('items', []))}")
        
        if response.get('items'):
            print("\nSample video:")
            item = response['items'][0]
            print(f"Title: {item['snippet']['title']}")
            print(f"Channel: {item['snippet']['channelTitle']}")
            print(f"Video ID: {item['id']['videoId']}")
        
        return True
        
    except Exception as e:
        print(f"\nAPI Test Failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure YOUTUBE_API_KEY is set correctly")
        print("2. Check if YouTube Data API v3 is enabled in Google Cloud Console")
        print("3. Verify your API key has not expired")
        return False

def scrape_specific_videos(video_ids):
    """Scrape comments from specific video IDs"""
    youtube = create_youtube_service()
    all_comments = []
    
    for video_id in video_ids:
        video_meta = get_videos_metadata(youtube, [video_id]).get(video_id, {})
        if video_meta:
            comments = get_video_comments(youtube, video_id, video_meta, max_comments=50)
            all_comments.extend(comments)
            time.sleep(1)
    
    return all_comments

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("="*60)
    print("YouTube AI Comment Scraper")
    print("="*60)
    print(f"Date range: {start_date[:10]} to {end_date[:10]}")
    print(f"Comment length: {MIN_WORDS} to {MAX_WORDS} words")
    print("="*60)
    
    # Test API first
    print("\nTesting YouTube API connection...")
    if not test_youtube_api():
        exit(1)
    
    print("\n" + "="*60)
    print("Starting main scraping...")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # OPTION 1: Use predefined queries
        comments_data = scrape_youtube_ai_data(
            queries=[
                "AI bubble",
                "AI regulation",
                "AI ethics debate",
                "artificial intelligence future",
                "AI job impact",
                "why is AI important"
                "AI risks benefits"
            ],
            videos_per_query=5,
            comments_per_video=100
        )
        
        # Create DataFrame
        if comments_data:
            df = pd.DataFrame(comments_data)
            
            # Add sentiment label
            df['sentiment_label'] = df['sentiment_polarity'].apply(
                lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
            )
            
            # Save to CSV
            output_file = "youtube_ai.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            # Print summary
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print("SCRAPING COMPLETE!")
            print(f"{'='*60}")
            print(f"Total comments collected: {len(df)}")
            print(f"Unique videos: {df['video_id'].nunique()}")
            print(f"Average comment length: {df['comment_length'].mean():.1f} words")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"\nSentiment distribution:")
            print(df['sentiment_label'].value_counts())
            print(f"\nTop channels:")
            print(df['video_channel'].value_counts().head())
            print(f"\nData saved to: {output_file}")
            
            # Show sample
            print(f"\nSample data (first 3 comments):")
            for i, row in df.head(3).iterrows():
                print(f"\n{i+1}. [{row['sentiment_label'].upper()}] {row['text'][:100]}...")
                print(f"   Length: {row['comment_length']} words | Likes: {row['likes']} | Channel: {row['video_channel'][:30]}")
            
        else:
            print("\nNo relevant comments collected!")
            print("\nSuggestions:")
            print("1. Try broader search queries")
            print("2. Increase videos_per_query and comments_per_video")
            print("3. Check if your API key has enough quota")
            print("4. Try specific video IDs (uncomment OPTION 2 in code)")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Check your API key: echo $YOUTUBE_API_KEY")
        print("2. Verify API key is valid and YouTube Data API v3 is enabled")
        print("3. Check your API quota at: https://console.cloud.google.com/apis/dashboard")