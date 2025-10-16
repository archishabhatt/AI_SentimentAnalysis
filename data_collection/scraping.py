import praw
import pandas as pd
import time
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
    "MachineLearning", "deeplearning", "LocalLLaMA", "ChatGPT", "artificial", "singularity",
    "datascience", "MLQuestions", "GPT3", "AI_Agents", "aiengineering", "antiai", "Futurology"
]

# --- Scraping settings ---
post_limit = 10000
min_comments = 1

# --- Keywords ---
ai_keywords = [
    "artificial intelligence", "AI", "GPT","ChatGPT","Deepseek",
    "LLM", "AI ethics", "generative AI", "Large Language Model"
]

opinion_keywords = [
    "think", "believe", "feel", "opinion", "perspective", "advantage", "disadvantage",
    "experience", "should", "ethical", "concern", "problem", "impact",
    "bias", "responsibility", "society","debate", "discuss", "argue"
]

ethical_keywords = [
    "ethical", "concern", "impact", "bias", "responsibility", "society"
]

# --- Helper functions ---
def contains_ai(text):
    text = text.lower()
    return any(k.lower() in text for k in ai_keywords)

def contains_opinion(text):
    text = text.lower()
    return any(k.lower() in text for k in opinion_keywords)

def contains_ethical(text):
    text = text.lower()
    return any(k.lower() in text for k in ethical_keywords)

def is_subjective(text, threshold=0.3):
    return TextBlob(text).sentiment.subjectivity > threshold

def is_highly_opinionated_ai(text):
    return contains_ai(text) and contains_opinion(text) and is_subjective(text)

def is_ethical_ai_post(text):
    return is_highly_opinionated_ai(text) and contains_ethical(text)

# --- Storage ---
posts_data = []

# --- Scraping loop ---
for sub in subreddits:
    try:
        subreddit = reddit.subreddit(sub)
        print(f"Scraping subreddit: {sub}")
        
        for post in subreddit.new(limit=post_limit):
            try:
                if not post.is_self:
                    continue
                
                full_text = (post.title or "") + " " + (post.selftext or "")
                
                # Filter: opinionated AI + ethical/societal keywords + discussion + minimum words
                if len(full_text.split()) > 15 and post.num_comments >= min_comments and is_ethical_ai_post(full_text):
                    sentiment = TextBlob(full_text).sentiment
                    posts_data.append({
                        "subreddit": sub,
                        "title": post.title,
                        "selftext": post.selftext,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "id": post.id,
                        "url": post.url,
                        "created_utc": post.created_utc,
                        "num_comments": post.num_comments,
                        "post_length": len(full_text.split()),
                        "title_length": len(post.title.split()),
                        "sentiment_polarity": sentiment.polarity,
                        "sentiment_subjectivity": sentiment.subjectivity
                    })
                    
            except Exception as e:
                print(f"Skipped post {post.id} due to error: {e}")
                continue
        
        time.sleep(5)
                
    except Exception as e:
        print(f"Skipped subreddit {sub} due to error: {e}")
        continue

# --- Save to CSV ---
df = pd.DataFrame(posts_data)
df.to_csv("reddit_ai.csv", index=False)
print(f"Scraping finished! {len(df)} posts saved to reddit_ai.csv")
