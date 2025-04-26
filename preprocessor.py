# preprocessor.py
import pandas as pd
import re
import streamlit as st

@st.cache_data
def preprocess(data):
    """
    Preprocess the WhatsApp chat data into a structured DataFrame.
    The function is cached to avoid recomputation when the same data is uploaded.
    """
    # Detect the date format pattern
    if re.search('\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}\s\w{2}\s-\s', data):
        # MM/DD/YY format
        pattern = '\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}\s\w{2}\s-\s'
        date_format = '%d/%m/%y, %I:%M %p - '
    elif re.search('\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s', data):
        # DD/MM/YYYY format without AM/PM
        pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
        date_format = '%d/%m/%Y, %H:%M - '
    else:
        # Default to common format
        pattern = '\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}\s\w{2}\s-\s'
        date_format = '%d/%m/%y, %I:%M %p - '

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    
    try:
        # Try to parse dates with the detected format
        df['message_date'] = pd.to_datetime(df['message_date'], format=date_format)
    except ValueError:
        # If fails, try alternative formats
        try:
            df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')
        except ValueError:
            # As a last resort, try the flexible parser
            df['message_date'] = pd.to_datetime(df['message_date'], errors='coerce')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract users and messages
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract date features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create time periods
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    return df