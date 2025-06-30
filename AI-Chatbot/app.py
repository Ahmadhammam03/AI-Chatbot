from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import sys
import threading
import time
import requests
import zipfile

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 💾 Download model from Google Drive if not found
def download_and_extract_model():
    model_dir = './models/galaxy_alibaba_chatbot'
    if os.path.exists(model_dir):
        print("✅ Model already exists. Skipping download.")
        return

    zip_url = "https://drive.google.com/uc?export=download&id=1f6z0Tf61SUOnBaBEwNQgASfRLo2c2LC7"
    zip_path = "models.zip"

    print("📦 Downloading model from Google Drive...")
    r = requests.get(zip_url)
    with open(zip_path, 'wb') as f:
        f.write(r.content)
    print("📂 Extracting model files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./models')
    print("✅ Model downloaded and extracted.")

download_and_extract_model()

from utils.similarity_checker import SimilarityChecker
from utils.response_generator import ResponseGenerator

app = Flask(__name__)
CORS(app)

MODEL_PATH = './models/galaxy_alibaba_chatbot'
QA_DATABASE_PATH = './data/processed_data.json'

print("Initializing chatbot components...")
similarity_checker = SimilarityChecker()
response_generator = ResponseGenerator(MODEL_PATH)

with open(QA_DATABASE_PATH, 'r', encoding='utf-8') as f:
    qa_database = json.load(f)
print(f"✅ Loaded {len(qa_database)} Q&A pairs")

RELEVANT_KEYWORDS = [
    'galaxy', 'organisation', 'organization', 'alibaba', 'cloud',
    'certification', 'jordan', 'amman', 'recycling', 'empowerment',
    'training', 'aca', 'acp', 'ace', 'nonprofit', 'women', 'kids'
]

TRAINED_GREETINGS = [
    'hi', 'hello', 'good morning', 'hey', 'good afternoon', 
    'hi there', 'good evening', 'morning', 'hello!', 'hey there',
    'greetings', 'hi chatbot', "what's up", 'hi, i need help', 'good day'
]

THANK_YOU_MESSAGES = [
    'thank you', 'thanks', 'thank you so much', 'thanks a lot',
    'thank you very much', 'appreciate it', 'thanks for your help',
    'thank you for the information', 'thanks for helping', 'grateful',
    'thank you for the help', 'thanks for the info', 'much appreciated',
    'thank you!', 'thanks!', 'thx', 'ty', 'cheers'
]

def is_special_message(message):
    message_lower = message.lower().strip()
    return message_lower in TRAINED_GREETINGS or any(phrase in message_lower for phrase in THANK_YOU_MESSAGES)

def generate_with_timeout(response_generator, user_message, similar_qa, timeout=45):
    result = [None]
    exception = [None]
    def target():
        try:
            result[0] = response_generator.generate_response(user_message, similar_qa)
        except Exception as e:
            exception[0] = e
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive() or exception[0]:
        print("⚠️ Generation failed or timed out")
        return None
    return result[0]

def get_fallback_response(user_message, similar_qa):
    if similar_qa and len(similar_qa) > 0 and similar_qa[0][1] > 0.6:
        return similar_qa[0][0]['completion']
    question_lower = user_message.lower()
    if any(word in question_lower for word in ['galaxy', 'organisation', 'organization']):
        return "Galaxy Organisation is a nonprofit organization based in Jordan that focuses on empowerment through technology training and cloud certifications."
    elif any(word in question_lower for word in ['location', 'where', 'based', 'located']):
        return "Galaxy Organisation is located in Amman, Jordan."
    elif any(word in question_lower for word in ['alibaba', 'cloud']):
        return "Alibaba Cloud is a leading cloud service provider..."
    elif any(word in question_lower for word in ['certification', 'aca', 'acp', 'ace']):
        return "Galaxy Organisation provides Alibaba Cloud certification training..."
    elif any(word in question_lower for word in ['training', 'program', 'course']):
        return "Galaxy Organisation offers various training programs..."
    elif any(word in question_lower for word in ['contact', 'reach', 'phone', 'email']):
        return "You can contact Galaxy Organisation through their official channels."
    else:
        return "I specialize in Galaxy Organisation and Alibaba Cloud—please ask about programs or services."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'response': 'Please ask a question.', 'status': 'error'})

        print(f"📝 Processing: {user_message}")
        is_special = is_special_message(user_message)
        if not is_special and not similarity_checker.is_relevant_question(user_message.lower(), RELEVANT_KEYWORDS):
            return jsonify({
                'response': 'Please ask about Galaxy Organisation or Alibaba services.',
                'status': 'irrelevant'
            })

        similar_qa = similarity_checker.find_similar_qa(user_message, qa_database, top_k=3, threshold=0.7)
        force_generation = is_special
        if not force_generation and similar_qa and similar_qa[0][1] > 0.9:
            response = similar_qa[0][0]['completion']
            status = 'exact_match'
        else:
            response = generate_with_timeout(response_generator, user_message, similar_qa, timeout=10)
            if response is None or len(response.strip()) < 5:
                response = get_fallback_response(user_message, similar_qa)
                status = 'fallback'
            else:
                status = 'generated'

        processing_time = round(time.time() - start_time, 2)
        return jsonify({
            'response': response,
            'status': status,
            'confidence': similar_qa[0][1] if similar_qa else 0.0,
            'processing_time': processing_time
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'response': 'Sorry, something went wrong.',
            'status': 'error',
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'qa_pairs': len(qa_database)
    })

if __name__ == '__main__':
    print("🚀 Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
