from flask import Flask, request, jsonify
from agents.pdf_agent import PDFScrapingAgent
from agents.web_agent import WebsiteScrapingAgent
from agents.preprocessing_agent import PreprocessingAgent
from datasets.qna_generator import QnADatasetGenerator
from datasets.chain_generator import ChainOfThoughtDatasetGenerator
from datasets.convo_generator import ConversationalDatasetGenerator
from utils import save_csv
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Initialize agents
pdf_agent = PDFScrapingAgent()
web_agent = WebsiteScrapingAgent()
preprocessing_agent = PreprocessingAgent()
qna_generator = QnADatasetGenerator()
chain_generator = ChainOfThoughtDatasetGenerator()
convo_generator = ConversationalDatasetGenerator()

@app.route('/scrape_pdf', methods=['POST'])
def scrape_pdf():
    file = request.files['file']
    raw_text = pdf_agent.scrape(file)
    cleaned_text = preprocessing_agent.clean(raw_text)
    return jsonify({'cleaned_text': cleaned_text})

@app.route('/scrape_url', methods=['POST'])
def scrape_url():
    url = request.json['url']
    raw_text = web_agent.scrape(url)
    cleaned_text = preprocessing_agent.clean(raw_text)
    return jsonify({'cleaned_text': cleaned_text})

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    data = request.json
    cleaned_text = data['cleaned_text']
    dataset_type = data['dataset_type']

    if dataset_type == "QnA":
        result_data = qna_generator.generate(cleaned_text)
    elif dataset_type == "Chain-of-Thought":
        result_data = chain_generator.generate(cleaned_text)
    elif dataset_type == "Conversational":
        result_data = convo_generator.generate(cleaned_text)

    file_name = save_csv(result_data, dataset_type)
    return jsonify({'file_name': file_name})

if __name__ == '__main__':
    app.run(debug=True)
