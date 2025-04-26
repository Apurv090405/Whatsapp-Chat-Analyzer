# helper.py
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from urlextract import URLExtract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import timedelta

nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained models
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


extract = URLExtract()

def fetch_stats(selected_user,df):

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

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message']
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):


    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def intent_detection(selected_user, df):
    # This is a placeholder. In a real-world scenario, you'd use a pre-trained model.
    intents = ['question', 'statement', 'command', 'request']
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['intent'] = df['message'].apply(lambda x: np.random.choice(intents))
    return df['intent'].value_counts()


def toxicity_detection(selected_user, df):
    # This is a placeholder. In a real-world scenario, you'd use a pre-trained model.
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['toxicity'] = df['message'].apply(lambda x: np.random.random())  # Random toxicity score
    return df['toxicity'].mean()


def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    results = []
    for message in df['message']:
        result = sentiment_model(message[:512])[0]  # BERT models typically have a max length of 512 tokens
        score = int(result['label'].split()[0])  # Extract the numeric score
        sentiment = 'very negative' if score == 1 else 'negative' if score == 2 else 'neutral' if score == 3 else 'positive' if score == 4 else 'very positive'
        results.append({'message': message, 'sentiment': sentiment, 'score': result['score']})

    return pd.DataFrame(results)

def emotion_detection(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    results = []
    for message in df['message']:
        emotion_scores = emotion_model(message[:512])[0]  # BERT models typically have a max length of 512 tokens
        emotion = max(emotion_scores, key=lambda x: x['score'])
        results.append({'message': message, 'emotion': emotion['label'], 'score': emotion['score']})

    return pd.DataFrame(results)


def generate_summary(df, selected_user):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    full_text = " ".join(df['message'].tolist())
    inputs = bart_tokenizer([full_text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=100,
                                      early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary


def predict_continuation(df, selected_user):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    last_messages = df['message'].tail(5).tolist()
    input_text = " ".join(last_messages)
    inputs = bart_tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
    continuation_ids = bart_model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=100,
                                           early_stopping=True)
    continuation = bart_tokenizer.decode(continuation_ids[0], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
    return continuation


def analyze_tone(df, selected_user):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    tones = ['Empathetic', 'Sarcastic', 'Serious', 'Lighthearted']
    tone_scores = [0, 0, 0, 0]

    for message in df['message']:
        result = sentiment_model(message[:512])[0]
        score = int(result['label'].split()[0])
        if score <= 2:
            tone_scores[1] += 1  # Sarcastic
        elif score == 3:
            tone_scores[2] += 1  # Serious
        else:
            tone_scores[0] += 1  # Empathetic

        emotion = emotion_detection(selected_user, pd.DataFrame({'message': [message]}))['emotion'].values[0]
        if emotion in ['joy', 'surprise']:
            tone_scores[3] += 1  # Lighthearted

    total = sum(tone_scores)
    tone_scores = [score / total for score in tone_scores]

    return tones, tone_scores



def create_sentiment_wordcloud(df, selected_user):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    words = []
    sentiments = {}

    for message in df['message']:
        words.extend(message.lower().split())
        result = sentiment_model(message[:512])[0]
        score = int(result['label'].split()[0])
        sentiment = 'negative' if score <= 2 else 'neutral' if score == 3 else 'positive'
        for word in message.lower().split():
            sentiments[word] = sentiment

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))

    return wordcloud, sentiments
