from flask import Flask, jsonify, render_template
from flask_cors import CORS
import random
from transformers import pipeline
import threading
import time
import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the templates folder
template_dir = os.path.join(script_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)
CORS(app)

# Load the text generation pipeline with DistilGPT2
generator = pipeline('text-generation', model='distilgpt2', device=-1)  # device=-1 uses CPU

# General random topics
topics = [
    "AI getting conciousness and taking over whole universe", "how to contact aliens"
]

conversation = []
last_fetched_index = 0

def generate_response(prompt, max_length=250):
    response = generator(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']
    return response.strip()

class Chatbot:
    def __init__(self, name):
        self.name = name

    def ask_question(self):
        topic = random.choice(topics)
        return f"let us talk about {topic}?"

    def generate_response(self, prompt):
        return generate_response(prompt)

def chat_round(bot1, bot2):
    global conversation
    
    question = bot1.ask_question()
    conversation.append({'chatbot': bot1.name, 'message': question})
    
    answer = bot2.generate_response(question)
    conversation.append({'chatbot': bot2.name, 'message': answer})
    
    follow_up = bot2.ask_question()
    conversation.append({'chatbot': bot2.name, 'message': follow_up})
    
    final_answer = bot1.generate_response(follow_up)
    conversation.append({'chatbot': bot1.name, 'message': final_answer})
    
    # Keep only the last 100 messages
    if len(conversation) > 1000:
        conversation = conversation[-100:]

def endless_conversation():
    bot1 = Chatbot("Bot 1")
    bot2 = Chatbot("Bot 2")
    
    while True:
        chat_round(bot1, bot2)
        time.sleep(10)  # Wait for 10 seconds before the next round

# Start the endless conversation in a separate thread
threading.Thread(target=endless_conversation, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['GET'])
def api_chat():
    global last_fetched_index
    if last_fetched_index > len(conversation):
        last_fetched_index = max(0, len(conversation) - 100)
    new_messages = conversation[last_fetched_index:]
    last_fetched_index = len(conversation)
    return jsonify(new_messages)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
