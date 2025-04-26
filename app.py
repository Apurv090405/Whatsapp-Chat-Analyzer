# app.py
import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# Set page config for wider layout
st.set_page_config(layout="wide", page_title="WhatsApp Chat Analyzer", page_icon="💬")

# Custom CSS for different colors and better fonts
st.markdown("""
    <style>
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    h1 { color: #4CAF50; }
    h2 { color: #FF5722; }
    h3 { color: #2196F3; }
    .header { font-size:24px; color: #FF5722; }
    .stats-header { font-size:22px; color: #4CAF50; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Title
st.sidebar.title("💬 WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt"])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Analyze for", user_list)

    if st.sidebar.button("Show Analysis"):

        # Top Statistics Section with colors and font-size enhancements
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.markdown("<div class='stats-header'>🔢 Top Statistics</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Total Messages")
            st.title(f"📩 {num_messages}")
        with col2:
            st.subheader("Total Words")
            st.title(f"📝 {words}")
        with col3:
            st.subheader("Media Shared")
            st.title(f"📷 {num_media_messages}")
        with col4:
            st.subheader("Links Shared")
            st.title(f"🔗 {num_links}")

        # Monthly Timeline
        st.markdown("<h3>📅 Monthly Timeline</h3>", unsafe_allow_html=True)
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.markdown("<h3>📆 Daily Timeline</h3>", unsafe_allow_html=True)
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.markdown("<h3>📊 Activity Map</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.subheader("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Wordcloud
        st.markdown("<h3>☁️ Wordcloud</h3>", unsafe_allow_html=True)
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common Words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.markdown("<h3>📜 Most Common Words</h3>", unsafe_allow_html=True)
        st.pyplot(fig)

        # Sentiment Analysis
        st.markdown("<h3>😊 Sentiment Analysis</h3>", unsafe_allow_html=True)
        sentiment_df = helper.sentiment_analysis(selected_user, df)
        st.write("Overall Sentiment Distribution:")
        fig, ax = plt.subplots()
        sentiment_df['sentiment'].value_counts().plot(kind='bar')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        st.write("Sample Messages with Sentiment:")
        st.dataframe(sentiment_df.sample(10))

        # Intent Detection
        st.markdown("<h3>🔍 Intent Detection</h3>", unsafe_allow_html=True)
        intents = helper.intent_detection(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(intents.index, intents.values)
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Emotion Detection
        st.markdown("<h3>😄 Emotion Detection</h3>", unsafe_allow_html=True)
        emotion_df = helper.emotion_detection(selected_user, df)
        st.write("Overall Emotion Distribution:")
        fig, ax = plt.subplots()
        emotion_df['emotion'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        st.pyplot(fig)
        st.write("Sample Messages with Detected Emotions:")
        st.dataframe(emotion_df.sample(10))

        # Toxicity Detection
        st.markdown("<h3>⚠️ Toxicity Detection</h3>", unsafe_allow_html=True)
        toxicity = helper.toxicity_detection(selected_user, df)
        st.write(f"Average toxicity score: {toxicity:.2f} (0 is non-toxic, 1 is very toxic)")

 # Summarization
        st.markdown("<h3>📝 Conversation Summary</h3>", unsafe_allow_html=True)
        summary = helper.generate_summary(df, selected_user)
        st.write(summary)

        # Predictive Conversation Continuation
        st.markdown("<h3>🔮 Predictive Conversation Continuation</h3>", unsafe_allow_html=True)
        continuation = helper.predict_continuation(df, selected_user)
        st.write("Predicted continuation:")
        st.write(continuation)

        # Tone of Voice Visualization
        st.markdown("<h3>🎭 Tone of Voice Visualization</h3>", unsafe_allow_html=True)
        tones, tone_scores = helper.analyze_tone(df, selected_user)
        fig = go.Figure(data=[go.Bar(x=tones, y=tone_scores, marker_color=['blue', 'red', 'green', 'yellow'])])
        fig.update_layout(title_text='Conversation Tone Analysis')
        st.plotly_chart(fig)

        # Interactive Wordcloud with Sentiment Overlay
        st.markdown("<h3>☁️ Interactive Wordcloud with Sentiment</h3>", unsafe_allow_html=True)
        wordcloud, word_sentiments = helper.create_sentiment_wordcloud(df, selected_user)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        st.write("Word Sentiments:")
        st.write(word_sentiments)