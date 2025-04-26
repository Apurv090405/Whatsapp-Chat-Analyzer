# helper.py
import pandas as pd
from collections import Counter
import os
import streamlit as st
from wordcloud import WordCloud
from urlextract import URLExtract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
import time

# Download NLTK data at first run
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Initialize models with caching to avoid reloading
@st.cache_resource
def initialize_models():
    # Download NLTK data
    download_nltk_data()
    
    # Load models with memory efficiency in mind
    # Use smaller models when possible
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", framework="tf")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, framework="tf")
    
    return sentiment_model, emotion_model

# Initialize global variables
extract = URLExtract()
sentiment_model = None
emotion_model = None

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(" ".join(temp['message']))
    return df_wc

def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    
    # Use NLTK stopwords to filter common words
    stop_words = set(stopwords.words('english'))

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # Limit to top 20 for better visualization
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def intent_detection(selected_user, df):
    # Simplified intent detection using keywords instead of a heavy model
    intents = ['question', 'statement', 'command', 'request']
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Basic keyword-based intent detection
    def detect_intent(message):
        message = message.lower()
        if '?' in message or message.startswith(('what', 'why', 'how', 'when', 'where')):
            return 'question'
        elif message.startswith(('please', 'could you', 'would you')):
            return 'request'
        elif message.startswith(('do', 'go', 'make', 'let')):
            return 'command'
        else:
            return 'statement'
    
    df['intent'] = df['message'].apply(detect_intent)
    return df['intent'].value_counts()

def toxicity_detection(selected_user, df):
    # Simplified toxicity detection using keywords
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    toxic_words = ['hate', 'stupid', 'idiot', 'dumb', 'fool', 'damn', 'shit', 'crap', 'ass']
    
    def calculate_toxicity(message):
        message = message.lower()
        count = sum(1 for word in toxic_words if word in message.split())
        # Normalize to 0-1 range
        return min(count / 3, 1.0)  # Cap at 1.0
    
    df['toxicity'] = df['message'].apply(calculate_toxicity)
    return df['toxicity'].mean()

def sentiment_analysis(selected_user, df):
    global sentiment_model
    if sentiment_model is None:
        sentiment_model, _ = initialize_models()
        
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Sample only a subset of messages for performance on large datasets
    sample_size = min(200, len(df))
    sampled_df = df.sample(sample_size)
    
    results = []
    for message in sampled_df['message']:
        if len(message) > 5:  # Only analyze messages with actual content
            try:
                result = sentiment_model(message[:512])[0]  # BERT models typically have a max length of 512 tokens
                score = int(result['label'].split()[0])  # Extract the numeric score
                sentiment = 'very negative' if score == 1 else 'negative' if score == 2 else 'neutral' if score == 3 else 'positive' if score == 4 else 'very positive'
                results.append({'message': message, 'sentiment': sentiment, 'score': result['score']})
            except Exception as e:
                # Skip if there's an error in sentiment analysis
                continue
                
    return pd.DataFrame(results) if results else pd.DataFrame(columns=['message', 'sentiment', 'score'])

def emotion_detection(selected_user, df):
    global emotion_model
    if emotion_model is None:
        _, emotion_model = initialize_models()
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Sample only a subset of messages for performance on large datasets
    sample_size = min(200, len(df))
    sampled_df = df.sample(sample_size)
    
    results = []
    for message in sampled_df['message']:
        if len(message) > 5:  # Only analyze messages with actual content
            try:
                emotion_scores = emotion_model(message[:512])[0]  # BERT models typically have a max length of 512 tokens
                emotion = max(emotion_scores, key=lambda x: x['score'])
                results.append({'message': message, 'emotion': emotion['label'], 'score': emotion['score']})
            except Exception as e:
                # Skip if there's an error in emotion analysis
                continue
                
    return pd.DataFrame(results) if results else pd.DataFrame(columns=['message', 'emotion', 'score'])

def generate_summary(df, selected_user):
    # Simplified summary generator using basic statistics instead of a heavy model
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Get basic statistics
    total_messages = len(df)
    total_words = sum(len(message.split()) for message in df['message'])
    avg_message_length = total_words / total_messages if total_messages > 0 else 0
    time_span = (df['date'].max() - df['date'].min()).days + 1
    
    # Most active times
    most_active_day = df['day_name'].value_counts().idxmax()
    most_active_hour = df['hour'].value_counts().idxmax()
    
    # Create a summary
    summary = f"This conversation contains {total_messages} messages over {time_span} days. "
    summary += f"The average message length is {avg_message_length:.1f} words. "
    summary += f"The most active day is {most_active_day} and the most active hour is around {most_active_hour}:00. "
    
    # Add common topics (most common words)
    common_words = Counter()
    stop_words = set(stopwords.words('english'))
    for message in df['message']:
        common_words.update([w for w in message.lower().split() if w not in stop_words and len(w) > 3])
    
    top_topics = ", ".join([word for word, _ in common_words.most_common(5)])
    summary += f"Common topics discussed include: {top_topics}."
    
    return summary

def predict_continuation(df, selected_user):
    # Simplified continuation prediction
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    if len(df) < 5:
        return "Not enough messages to predict continuation."
    
    # Get the last few messages
    last_messages = df['message'].tail(5).tolist()
    users = df['user'].tail(5).tolist()
    
    # Simple pattern-based prediction
    greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    question_patterns = ['?', 'what', 'how', 'when', 'why', 'where', 'who']
    
    last_msg = last_messages[-1].lower()
    
    if any(pattern in last_msg for pattern in greeting_patterns):
        return f"Based on the greeting pattern, {users[-1]} might continue with pleasantries or ask about well-being."
    elif any(pattern in last_msg for pattern in question_patterns):
        return f"A question was asked. Expect {users[-2] if users[-1] != users[-2] else 'someone'} to provide an answer or clarification."
    elif '<Media omitted>' in last_msg:
        return "A media file was shared. Expect comments or reactions to the shared media."
    else:
        # Generic continuation
        return f"The conversation might continue with {users[-2] if users[-1] != users[-2] else 'another user'} responding to the last message or introducing a new topic."

def analyze_tone(df, selected_user):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    tones = ['Empathetic', 'Sarcastic', 'Serious', 'Lighthearted']
    tone_scores = [0, 0, 0, 0]

    # Simple keyword-based tone detection
    empathetic_keywords = ['understand', 'feel', 'sorry', 'hope', 'care', 'love', 'appreciate']
    sarcastic_keywords = ['yeah right', 'sure', 'whatever', 'oh really', 'brilliant', 'genius']
    serious_keywords = ['important', 'serious', 'focus', 'consider', 'must', 'need', 'should']
    lighthearted_keywords = ['lol', 'haha', '😂', '😄', 'joke', 'funny', 'lmao', 'rofl']

    for message in df['message']:
        message_lower = message.lower()
        
        # Check for keywords
        if any(keyword in message_lower for keyword in empathetic_keywords):
            tone_scores[0] += 1
        if any(keyword in message_lower for keyword in sarcastic_keywords):
            tone_scores[1] += 1
        if any(keyword in message_lower for keyword in serious_keywords):
            tone_scores[2] += 1
        if any(keyword in message_lower for keyword in lighthearted_keywords):
            tone_scores[3] += 1

    # Ensure we have at least some tone detected
    if sum(tone_scores) == 0:
        # Default distribution if no keywords detected
        tone_scores = [0.25, 0.25, 0.25, 0.25]
    else:
        # Normalize to get percentages
        total = sum(tone_scores)
        tone_scores = [score / total for score in tone_scores]

    return tones, tone_scores

def create_sentiment_wordcloud(df, selected_user):
    global sentiment_model
    if sentiment_model is None:
        sentiment_model, _ = initialize_models()
        
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Sample for better performance
    sample_size = min(100, len(df))
    sampled_df = df.sample(sample_size)

    words = []
    sentiments = {}

    # Get stopwords from NLTK
    stop_words = set(stopwords.words('english'))

    for message in sampled_df['message']:
        # Skip media messages and filter stopwords
        if message != '<Media omitted>\n':
            filtered_words = [word.lower() for word in message.split() 
                              if word.lower() not in stop_words and len(word) > 3]
            words.extend(filtered_words)
            
            # Only analyze longer messages for sentiment
            if len(message) > 10:
                try:
                    result = sentiment_model(message[:512])[0]
                    score = int(result['label'].split()[0])
                    sentiment = 'negative' if score <= 2 else 'neutral' if score == 3 else 'positive'
                    
                    for word in filtered_words:
                        sentiments[word] = sentiment
                except Exception:
                    # Skip if there's an error in sentiment analysis
                    continue

    # Create wordcloud with frequency weights
    if words:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    else:
        # Create an empty wordcloud if no words
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate("No data")

    return wordcloud, sentiments