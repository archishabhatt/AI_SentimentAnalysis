import praw
import pandas as pd
import time
from datetime import datetime, timezone
from textblob import TextBlob

# --- Reddit API credentials ---
reddit = praw.Reddit(
    client_id="ZJsFu16c3va0C51kYDtEuw",
    client_secret="VIUps0dC9u3n1DBltprGqJQOwM8daA",
    username="desultoryphilosopher",
    password="Archisha@123",
    user_agent="AI scraper by /u/desultoryphilosopher"
)

# --- Subreddits to scrape ---
subreddits = [
    "MachineLearning", "deeplearning", "LocalLLaMA", "ChatGPT", "artificial", 
    "singularity", "datascience", "GPT3", "AI_Agents", "aiengineering", 
    "Futurology", "technology", "science", "programming", "computerscience"
]

# --- Scraping settings ---
post_limit = 100
min_comments = 5
min_word_count = 10  # Minimum words in post+comments for good NLP
max_word_count = 1500  # Maximum words to avoid extremely long posts

# --- Date filtering ---
start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 12, 31, tzinfo=timezone.utc)

# --- Keywords for filtering ---
ai_keywords = [
    "artificial intelligence", "AI", "GPT", "ChatGPT", "DeepSeek", "Claude", "Gemini",
    "LLM", "Large Language Model", "machine learning", "neural network", "transformer",
    "AGI", "Artificial General Intelligence", "generative AI", "AI model", "AI system"
]

opinion_keywords = [
    "think", "believe", "feel", "opinion", "perspective", "view", "standpoint",
    "should", "would", "could", "might", "argument", "debate", "discussion",
    "pro", "con", "advantage", "disadvantage", "benefit", "drawback"
]

societal_keywords = [
    "ethical", "ethics", "moral", "society", "societal", "impact", "consequence",
    "future", "humanity", "job", "employment", "economy", "privacy", "security",
    "bias", "fair", "transparent", "accountable", "responsible", "regulation",
    "govern", "control", "risk", "danger", "opportunity", "potential"
]

# --- Helper functions ---
def contains_ai(text):
    """Check if text contains AI-related keywords"""
    text = text.lower()
    return any(keyword.lower() in text for keyword in ai_keywords)

def contains_opinion(text):
    """Check if text contains opinion indicators"""
    text = text.lower()
    return any(keyword.lower() in text for keyword in opinion_keywords)

def contains_societal(text):
    """Check if text contains societal/ethical keywords"""
    text = text.lower()
    return any(keyword.lower() in text for keyword in societal_keywords)

def get_word_count(text):
    """Get word count of text"""
    return len(str(text).split())

def is_relevant_post(text):
    """Check if post is relevant for our analysis"""
    word_count = get_word_count(text)
    
    # Check length requirements
    if word_count < min_word_count or word_count > max_word_count:
        return False
    
    # Check content requirements
    if not contains_ai(text):
        return False
    
    # Must contain either opinion or societal keywords
    if not (contains_opinion(text) or contains_societal(text)):
        return False
    
    # Must be subjective enough
    if TextBlob(text).sentiment.subjectivity < 0.3:
        return False
    
    return True

# --- Scrape comments from post ---
def scrape_post_comments(post):
    """Extract comments from a post"""
    comments_data = []
    
    try:
        # Expand all comments
        post.comments.replace_more(limit=2)
        
        for comment in post.comments.list():
            # Skip deleted/removed comments
            if not comment.body or comment.body in ['[deleted]', '[removed]']:
                continue
            
            comment_text = comment.body
            word_count = get_word_count(comment_text)
            
            # Filter comments by length and relevance
            if (10 <= word_count <= 300 and 
                contains_ai(comment_text) and 
                (contains_opinion(comment_text) or contains_societal(comment_text))):
                
                sentiment = TextBlob(comment_text).sentiment
                
                comments_data.append({
                    # --- TEXT & ANALYSIS (YouTube compatible) ---
                    "text": comment_text,
                    "comment_length": word_count,
                    "sentiment_polarity": sentiment.polarity,
                    "sentiment_subjectivity": sentiment.subjectivity,
                    
                    # --- ENGAGEMENT (YouTube compatible) ---
                    "likes": comment.score,
                    "num_replies": len(comment.replies),
                    
                    # --- TIME ---
                    "created_utc": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                    
                    # --- SOURCE IDENTIFIERS ---
                    "source": "reddit",
                    "source_id": f"reddit_comment_{comment.id}",
                    "post_id": post.id,
                    
                    # --- POST METADATA ---
                    "post_title": post.title,
                    "subreddit": str(post.subreddit),
                    "post_score": post.score,
                    "post_upvote_ratio": post.upvote_ratio,
                    "post_num_comments": post.num_comments,
                    "post_created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                    
                    # --- CATEGORY FLAGS ---
                    "contains_ai": contains_ai(comment_text),
                    "contains_opinion": contains_opinion(comment_text),
                    "contains_societal": contains_societal(comment_text)
                })
                
    except Exception as e:
        print(f"Error scraping comments from post {post.id}: {e}")
    
    return comments_data

# --- Main scraping function ---
def scrape_reddit_ai_data():
    """Main function to scrape Reddit AI discussions"""
    
    all_comments_data = []
    posts_processed = 0
    comments_collected = 0
    
    for sub_name in subreddits:
        try:
            print(f"\n{'='*60}")
            print(f"Scraping subreddit: r/{sub_name}")
            print(f"{'='*60}")
            
            subreddit = reddit.subreddit(sub_name)
            
            # Use search to find relevant posts with date filtering
            search_query = "(" + " OR ".join(ai_keywords[:5]) + ") AND (opinion OR ethical OR impact OR future)"
            
            for post in subreddit.search(search_query, sort='new', limit=post_limit):
                try:
                    # Check post date
                    post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    if not (start_date <= post_date <= end_date):
                        continue
                    
                    # Check if it's a text post
                    if not post.is_self:
                        continue
                    
                    # Combine title and selftext
                    full_text = (post.title or "") + " " + (post.selftext or "")
                    
                    # Check relevance criteria
                    if (post.num_comments >= min_comments and 
                        is_relevant_post(full_text)):
                        
                        posts_processed += 1
                        print(f"\nProcessing post {posts_processed}: {post.title[:80]}...")
                        print(f"Date: {post_date.date()}, Comments: {post.num_comments}")
                        
                        # Scrape comments from this post
                        post_comments = scrape_post_comments(post)
                        
                        if post_comments:
                            all_comments_data.extend(post_comments)
                            comments_collected += len(post_comments)
                            print(f"Added {len(post_comments)} comments (Total: {comments_collected})")
                        
                        # Rate limiting
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"Error processing post {post.id}: {e}")
                    continue
                
                # Stop if we have enough data
                if comments_collected >= 1000:
                    print(f"\nReached target of {comments_collected} comments. Stopping...")
                    break
            
            # Rate limiting between subreddits
            time.sleep(3)
            
            if comments_collected >= 1000:
                break
                
        except Exception as e:
            print(f"Error scraping subreddit r/{sub_name}: {e}")
            continue
    
    return all_comments_data

# --- Main execution ---
if __name__ == "__main__":
    print("Starting Reddit AI data scraping...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Minimum word count: {min_word_count}")
    print(f"Maximum word count: {max_word_count}")
    
    start_time = time.time()
    
    # Scrape data
    comments_data = scrape_reddit_ai_data()
    
    # Create DataFrame
    if comments_data:
        df = pd.DataFrame(comments_data)
        
        # Add sentiment label for easier analysis
        df['sentiment_label'] = df['sentiment_polarity'].apply(
            lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
        )
        
        # Save to CSV
        output_file = "reddit_ai.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Print summary
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("SCRAPING COMPLETE!")
        print(f"{'='*60}")
        print(f"Total comments collected: {len(df)}")
        print(f"Unique posts: {df['post_id'].nunique()}")
        print(f"Subreddits covered: {df['subreddit'].nunique()}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"\nSentiment distribution:")
        print(df['sentiment_label'].value_counts())
        print(f"\nTop subreddits:")
        print(df['subreddit'].value_counts().head())
        print(f"\nData saved to: {output_file}")
        
        # Show sample
        print(f"\nSample data:")
        print(df[['text', 'sentiment_label', 'subreddit', 'comment_length']].head())
        
    else:
        print("No data collected!")