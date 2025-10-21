# nlp_summarizer.py

import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

# ---------- STEP 1: Read and clean text ----------
file_path = "reviews50.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r'\s+', ' ', text.strip())

# ---------- STEP 2: Basic sentiment analysis ----------
blob = TextBlob(text)
sentiment_score = blob.sentiment.polarity

if sentiment_score > 0.2:
    sentiment_label = "Overall Positive"
elif sentiment_score < -0.2:
    sentiment_label = "Overall Negative"
else:
    sentiment_label = "Mixed/Neutral"

# ---------- STEP 3: Simple NLP-based summarization ----------
sentences = sent_tokenize(text)
stop_words = set(stopwords.words('english'))

# Word frequency
word_freq = {}
for word in word_tokenize(text.lower()):
    if word.isalpha() and word not in stop_words:
        word_freq[word] = word_freq.get(word, 0) + 1

# Sentence scoring
sentence_scores = {}
for sent in sentences:
    for word in word_tokenize(sent.lower()):
        if word in word_freq:
            sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

# Top sentences (simple extractive summary)
num_sentences = max(5, len(sentences) // 10)  # top 10%
sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
summary_sentences = sorted_sentences[:num_sentences]

final_summary = " ".join(summary_sentences)

# ---------- STEP 4: Display result ----------
print("\n===== TECH SUMMARIZER REPORT =====\n")
print(f"File Analyzed: {file_path}")
print(f"Sentiment Score: {sentiment_score:.2f}")
print(f"Overall Feedback: {sentiment_label}\n")

print("🔍 Detailed Summary:\n")
print(final_summary)

# ---------- STEP 5: Save to file ----------
with open("summary50.txt", "w", encoding="utf-8") as f:
    f.write("===== TECH SUMMARIZER REPORT =====\n")
    f.write(f"File Analyzed: {file_path}\n")
    f.write(f"Sentiment Score: {sentiment_score:.2f}\n")
    f.write(f"Overall Feedback: {sentiment_label}\n\n")
    f.write("🔍 Detailed Summary:\n")
    f.write(final_summary)

print("\n✅ Summary saved to 'summary50.txt'")
