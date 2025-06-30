from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import sys
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.similarity_checker import SimilarityChecker
    from utils.response_generator import ResponseGenerator
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Continue anyway for deployment

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = './models/galaxy_alibaba_chatbot'
QA_DATABASE_PATH = './data/processed_data.json'

# Azure timeout (shorter for cloud)
AZURE_TIMEOUT = 10

# Global variables
similarity_checker = None
response_generator = None
qa_database = None

def initialize_components():
    """Initialize components with error handling"""
    global similarity_checker, response_generator, qa_database
    
    try:
        logger.info("🚀 Initializing chatbot components...")
        
        # Initialize similarity checker
        similarity_checker = SimilarityChecker()
        logger.info("✅ Similarity checker loaded")
        
        # Initialize response generator
        response_generator = ResponseGenerator(MODEL_PATH)
        logger.info("✅ Response generator loaded")
        
        # Load Q&A database
        with open(QA_DATABASE_PATH, 'r', encoding='utf-8') as f:
            qa_database = json.load(f)
        logger.info(f"✅ Loaded {len(qa_database)} Q&A pairs")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        return False

# Keywords for relevance checking
RELEVANT_KEYWORDS = [
    'galaxy', 'organisation', 'organization', 'alibaba', 'cloud',
    'certification', 'jordan', 'amman', 'recycling', 'empowerment',
    'training', 'aca', 'acp', 'ace', 'nonprofit', 'women', 'kids'
]

# Trained greetings
TRAINED_GREETINGS = [
    'hi', 'hello', 'good morning', 'hey', 'good afternoon', 
    'hi there', 'good evening', 'morning', 'hello!', 'hey there',
    'greetings', 'hi chatbot', "what's up", 'hi, i need help', 'good day'
]

# Thank you messages
THANK_YOU_MESSAGES = [
    'thank you', 'thanks', 'thank you so much', 'thanks a lot',
    'thank you very much', 'appreciate it', 'thanks for your help',
    'thank you for the information', 'thanks for helping', 'grateful',
    'thank you for the help', 'thanks for the info', 'much appreciated',
    'thank you!', 'thanks!', 'thx', 'ty', 'cheers'
]

def is_special_message(message):
    """Check if message is greeting or thank you"""
    message_lower = message.lower().strip()
    
    if message_lower in TRAINED_GREETINGS:
        return True
    
    if any(phrase in message_lower for phrase in THANK_YOU_MESSAGES):
        return True
    
    return False

def generate_with_timeout(response_generator, user_message, similar_qa, timeout=AZURE_TIMEOUT):
    """Generate response with timeout"""
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
    
    if thread.is_alive():
        logger.warning(f"⚠️ Generation timed out after {timeout} seconds")
        return None
    
    if exception[0]:
        logger.warning(f"⚠️ Generation failed: {exception[0]}")
        return None
        
    return result[0]

def get_fallback_response(user_message, similar_qa):
    """Generate fallback response when model fails"""
    
    if similar_qa and len(similar_qa) > 0:
        best_match = similar_qa[0]
        if best_match[1] > 0.6:
            return best_match[0]['completion']
    
    question_lower = user_message.lower()
    
    if any(word in question_lower for word in ['galaxy', 'organisation', 'organization']):
        return "Galaxy Organisation is a nonprofit organization based in Jordan that focuses on empowerment through technology training and cloud certifications."
    
    elif any(word in question_lower for word in ['location', 'where', 'based', 'located']):
        return "Galaxy Organisation is located in Amman, Jordan."
    
    elif any(word in question_lower for word in ['alibaba', 'cloud']):
        return "Alibaba Cloud is a leading cloud service provider. Galaxy Organisation offers comprehensive training and certification programs for Alibaba Cloud services."
    
    elif any(word in question_lower for word in ['certification', 'aca', 'acp', 'ace']):
        return "Galaxy Organisation provides Alibaba Cloud certification training including Associate (ACA), Professional (ACP), and Expert (ACE) levels."
    
    elif any(word in question_lower for word in ['training', 'program', 'course']):
        return "Galaxy Organisation offers various training programs including Alibaba Cloud certifications and technology skills development."
    
    elif any(word in question_lower for word in ['contact', 'reach', 'phone', 'email']):
        return "You can contact Galaxy Organisation through their official channels for more information about programs and enrollment."
    
    else:
        return "I specialize in questions about Galaxy Organisation and Alibaba Cloud services. Could you please ask about our training programs, certifications, or services?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    
    # Check if components are initialized
    if not all([similarity_checker, response_generator, qa_database]):
        return jsonify({
            'response': 'Chatbot is starting up. Please try again in a moment.',
            'status': 'initializing'
        }), 503
    
    try:
        data = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({
                'response': 'Please ask a question about Galaxy Organisation or Alibaba.',
                'status': 'error'
            })

        logger.info(f"📝 Processing: {user_message}")

        # Check for special messages
        is_greeting = user_message.lower().strip() in TRAINED_GREETINGS
        is_thanks = any(phrase in user_message.lower().strip() for phrase in THANK_YOU_MESSAGES)
        is_special = is_greeting or is_thanks
        
        # Check relevance (skip for special messages)
        if not is_special and not similarity_checker.is_relevant_question(user_message.lower(), RELEVANT_KEYWORDS):
            return jsonify({
                'response': 'I specialize in questions about Galaxy Organisation and Alibaba. Please ask about their programs, services, or certifications.',
                'status': 'irrelevant'
            })

        # Find similar Q&As
        similar_qa = similarity_checker.find_similar_qa(
            user_message, qa_database, top_k=3, threshold=0.7
        )

        # Handle special messages
        if is_special:
            message_type = "greeting" if is_greeting else "thank you"
            logger.info(f"🎯 Special message: {message_type}")
            force_generation = True
        else:
            force_generation = False

        # Generate or use exact match
        if not force_generation and similar_qa and similar_qa[0][1] > 0.9:
            response = similar_qa[0][0]['completion']
            status = 'exact_match'
        else:
            logger.info("🤖 Generating response...")
            
            response = generate_with_timeout(
                response_generator, user_message, similar_qa
            )
            
            if response is None:
                logger.warning("⚠️ Using fallback")
                response = get_fallback_response(user_message, similar_qa)
                status = 'fallback'
            else:
                status = 'generated'

        # Ensure response exists
        if not response or len(response.strip()) < 5:
            response = get_fallback_response(user_message, similar_qa)
            status = 'fallback'

        processing_time = time.time() - start_time
        logger.info(f"⏱️ Processing time: {processing_time:.2f}s")

        return jsonify({
            'response': response,
            'status': status,
            'confidence': similar_qa[0][1] if similar_qa else 0.0,
            'processing_time': round(processing_time, 2)
        })

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error: {str(e)}")
        
        return jsonify({
            'response': 'Sorry, I encountered an error. Please try again.',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for Azure"""
    if not all([similarity_checker, response_generator, qa_database]):
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False
        }), 503
    
    return jsonify({
        'status': 'healthy', 
        'model_loaded': True,
        'qa_pairs': len(qa_database) if qa_database else 0
    })

# Initialize when starting
if __name__ == '__main__':
    success = initialize_components()
    if success:
        logger.info("🚀 Starting server...")
        # Use port from environment (Azure uses this)
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        logger.error("❌ Failed to start")
        # Start anyway with limited functionality
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)