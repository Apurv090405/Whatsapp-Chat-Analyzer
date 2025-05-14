# WCHAT: WhatsApp Analysis Tool

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Pandas](https://img.shields.io/badge/Pandas-v1.x-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.x-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.11.x-green)
![NLP](https://img.shields.io/badge/NLP-TextAnalysis-purple)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Enabled-yellow)

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [File Upload Format](#file-upload-format)
- [Preprocessing](#preprocessing)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction

**WCHAT** is a powerful WhatsApp analysis tool built using NLP and data visualization libraries. It enables users to gain deep insights into their WhatsApp chat history through visualizations, sentiment evaluation, intent and emotion detection, and conversation summarization. The application runs in a user-friendly web interface powered by **Streamlit**.

---

## Features

- ✅ Upload `.txt` file exported from WhatsApp chat.
- ✅ Displays **message statistics**: number of messages, words, media, and links.
- ✅ Generates:
  - 📅 Monthly Timeline
  - 📆 Daily Timeline
  - 🗺️ Activity Heatmap
- ✅ NLP-powered Insights:
  - ☁️ Wordcloud
  - 🗣️ Most Common Words
  - 😀 Sentiment Analysis
  - 🧠 Intent Detection
  - ❤️ Emotion Detection
  - ☢️ Toxicity Detection
  - 📄 Summarization
  - 🔊 Tone of Voice Visualization

---

## Model Architecture

The tool integrates various NLP modules to extract insights from text:
1. **Preprocessing Engine:** Cleans and structures raw WhatsApp chat data.
2. **Statistical Analyzer:** Extracts message count, words, media, and links.
3. **NLP Pipelines:**
   - Sentiment, Intent & Emotion detection using pretrained models.
   - Toxicity and summarization using transformers and rule-based scoring.
4. **Visualizer:** Generates timelines, wordclouds, and tone graphs using Matplotlib and Seaborn.

---

## File Upload Format

📂 Export your WhatsApp chat as `.txt` without media.

> Example:  
> ```
> 14/05/2023, 10:42 AM - Alice: Hello!  
> 14/05/2023, 10:43 AM - Bob: Hi there  
> ```

---

## Preprocessing

1. **Clean Text:** Remove system messages and irrelevant entries.
2. **Parse Dates:** Extract day, month, hour from each line.
3. **User Detection:** Separate group members’ names and their messages.
4. **Tokenization & Stopword Removal:** Prepare text for analysis.

---

## Screenshots
![image](https://github.com/user-attachments/assets/4da41689-86bb-4d58-8ef2-18e64320f802)

![image](https://github.com/user-attachments/assets/a848ae85-712e-4dcb-b11b-a5200a5f628c)

![image](https://github.com/user-attachments/assets/16e9b24a-5bf2-4477-aa81-8ae5d60022c5)

![image](https://github.com/user-attachments/assets/670578cf-e8fb-4cf0-988d-4fc6580bbb67)

![image](https://github.com/user-attachments/assets/2b8310bc-4aa7-47e7-93a3-ce5586b5a202)

---

## Installation

### Prerequisites
- Python >= 3.6

### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the file

``` streamlit run app.py ```

---

## Project Structure
```bash
├── app.py                # Main Streamlit app
├── preprocessor.py       # Data cleaning and parsing
├── helper.py             # NLP and plotting utilities
├── requirements.txt      # List of dependencies
├── results/              # Generated visualizations and summaries
└── README.md             # Project documentation
```

---


