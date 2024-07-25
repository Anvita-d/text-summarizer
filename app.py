from flask import Flask, render_template, request, jsonify
from summarizer import Summarizer
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

app = Flask(__name__)

nltk.download('punkt')

bert_model = Summarizer()

bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    method = data['method']

    if method == 'bert':
        summary = bert_summarize(text)
    elif method == 'transformers':
        summary = transformers_summarize(text)
    elif method == 'bart':
        summary = bart_summarize(text)
    elif method == 'extractive':
        summary = extractive_summarize(text)
    else:
        summary = "Invalid method"

    return jsonify({method + '_summary': summary})

def bert_summarize(text):
    return ''.join(bert_model(text, min_length=60))

def transformers_summarize(text):
    summarizer = pipeline("summarization")
    result = summarizer(text, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return result[0]['summary_text']

def bart_summarize(text):
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extractive_summarize(text):
    sentences = nltk.sent_tokenize(text)
    sentence_vectors = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(sentence_vectors, sentence_vectors)
    import networkx as nx
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    top_sentences = [sentence for _, sentence in ranked_sentences[:3]]
    summary = ' '.join(top_sentences)
    return summary

if __name__ == '__main__':
    app.run(debug=True)
