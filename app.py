# APP.PY

# FROM AND IMPORT CODES
import random
from difflib import SequenceMatcher
import json
from flask import Flask, render_template, redirect, session, request, jsonify
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import os
import pprint
import re
import sys
import json
import time
import string
import secrets
import tempfile
import subprocess
import select
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv

import logging
from logging import StreamHandler

from threading import Event
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, 
    send_from_directory, session, Response, stream_with_context
)
from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from difflib import SequenceMatcher, get_close_matches
from markupsafe import Markup
from docx import Document
from pydub import AudioSegment
import vosk
import wave
import pdfplumber
import uuid
import hashlib
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pty
import spacy
import psycopg2

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax


# Load environment variables from the visible file
load_dotenv('env.txt')

# Later in your imports, add the email service
from email_service import EmailService

flask_logger = logging.getLogger('werkzeug')
flask_logger.setLevel(logging.WARNING)  # Change to INFO if you want to capture more details


os.makedirs('uploads', exist_ok=True)

def initialize_bert():
    """Initialize BERT model and tokenizer"""
    global BERT_MODEL, BERT_TOKENIZER
    try:
        # Check model file details
        model_path = 'finra_compliance_model.pth'
        if os.path.exists(model_path):
            modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Loading model file: {model_path}")
            logger.info(f"Last modified: {modified_time}")
            logger.info(f"File size: {file_size:.2f} MB")
        
        # 1. Load the tokenizer from your saved tokenizer directory
        BERT_TOKENIZER = BertTokenizer.from_pretrained('finra_tokenizer')  # Path to your saved tokenizer

        # 2. Initialize the base model first
        BERT_MODEL = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # 3. Load your trained weights
        BERT_MODEL.load_state_dict(torch.load('finra_compliance_model.pth'))  # Path to your saved model

        # 4. Set to evaluation mode
        BERT_MODEL.eval()
        return True
    except Exception as e:
        logger.error(f"Error initializing BERT: {e}")
        return False

# Global variables for BERT model
BERT_MODEL = None
BERT_TOKENIZER = None

# Create a global email service instance
email_service = EmailService()

processed_files = {}  # Change from set() to dictionary


# Create a custom logger for your application
logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO)

# Ensure no duplicate handlers are added
if not logger.handlers:
    # Create handlers
    file_handler = logging.FileHandler('uploads/app.log')
    file_handler.setLevel(logging.INFO)

    console_handler = StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Suppress Flask's default logging to prevent duplicates
flask_logger = logging.getLogger('werkzeug')
flask_logger.setLevel(logging.WARNING)


# Function to clean the FINRA compliance text by removing embedded numbering
def clean_finra_analysis(text):

    # This will remove any leading numbers followed by a period (e.g., 1., 2., etc.)
    cleaned_text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    return cleaned_text


# Function for pdf text extraction by page
def extract_text_from_pdf(pdf_file_path):
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            page_text = []
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    page_text.append({"page": i, "text": text})  # Include page number and text
            return page_text
    except Exception as e:
        logger.info(f"Error extracting text from PDF: {e}")
        return []



def split_into_sentences(text):
    """Split text into individual sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


    
# Function for processing the response and ensuring only compliance issues are shown
def generate_compliance_analysis(analysis_results):

    # If no issues are found
    if not analysis_results or analysis_results == ["Compliance Check Completed. No issues found."]:
        return "Compliance Check Completed. No issues found."

    # Ensure each flagged instance is listed distinctly with conflict message and page number
    compliance_output = []
    for index, item in enumerate(analysis_results, start=1):  # Start numbering from 1

        # Each item is a dictionary with 'page' and 'response' keys
        page_number = item['page']

        # Remove redundant "Page Number: X" and hyphens from each line in the response
        issue_text = re.sub(r"- Page Number: \d+\n", "", item['response']).strip()
        issue_text = issue_text.replace("- Flagged Instance:", "Flagged Instance:").replace("- Specific Compliant Alternative:", "Specific Compliant Alternative:")

        # Add a blank line after each compliant alternative text (at the end of each instance)
        issue_text = re.sub(r"(Specific Compliant Alternative:.*?)(\n|$)", r"\1\n\n", issue_text, flags=re.DOTALL)

        # Extract flagged instance and compliant alternative from formatted text
        flagged_match = re.search(r'Flagged Instance: "(.*?)"', issue_text)
        alternative_match = re.search(r'Specific Compliant Alternative: "(.*?)"', issue_text)
        flagged_instance = flagged_match.group(1) if flagged_match else ""
        compliant_alternative = alternative_match.group(1) if alternative_match else "No specific compliant alternative available."

        # Format item with explicit keys for each part and include the current formatted output

        compliance_output.append({

            "page": page_number,
            "flagged_instance": flagged_instance,
            "compliant_alternative": compliant_alternative,
            "full_text": f"{index}. Potential Conflict Identified on Page {page_number}:\n{issue_text}"
        })

    return compliance_output

# Function to remove quotes from the analyzed text

def remove_quotes(text):
    """
    Remove all types of quotes (single, double) from the given text.
    """
    return re.sub(r'[\'"]', '', text)



# Usage of Prohibited Words in Check Compliance Function
def check_compliance(content_by_page, disclosures):
    finra_analysis = []
    logger.info("--- Starting Compliance Check ---")
    
    for page_content in content_by_page:
        page_num = page_content["page"]
        content = page_content["text"]
        cleaned_content = remove_quotes(content)
        
        # Perform compliance check using the centralized function
        flagged_instances = perform_compliance_check(cleaned_content, page_num)
        
        if flagged_instances:
            finra_analysis.extend(flagged_instances)
    
    return finra_analysis


def perform_compliance_check(text, page_num=None):
    """Checks text compliance using BERT for both flagging and generating alternatives."""
    try:
        global BERT_MODEL, BERT_TOKENIZER
        if BERT_MODEL is None or BERT_TOKENIZER is None:
            if not initialize_bert():
                raise Exception("Failed to initialize BERT model")

        # If text is empty, contains only whitespace, or is just a bullet point
        if not text or text.isspace() or text == "•" or text == "\u2022":
            return {"compliant": True, "flagged_instances": []}

        # If text is a dictionary, extract the 'text' field
        if isinstance(text, dict):
            text = text.get('text', '')
        
        # Clean the text
        text = re.sub(r'[•\u2022]', '', text)  # Remove bullet points
        text = re.sub(r'\s+', ' ', text).strip()  # Clean up whitespace
        
        # Split text into sentences
        sentences = split_into_sentences(text)
        flagged_instances = []

        # Load compliance examples database for alternatives
        compliance_examples = load_compliance_examples()

        # Process sentences in batches
        batch_size = 1024
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
        
            for sentence in batch:
                # Skip invalid sentences
                if not sentence.strip() or sentence.strip() in ['•', '\u2022']:
                    continue
            
                # Clean the sentence
                sentence = re.sub(r'[•\u2022]', '', sentence).strip()
                if not sentence:  # Skip if sentence becomes empty after cleaning
                    continue
                
                # Prepare input for BERT
                inputs = BERT_TOKENIZER(sentence, 
                                      return_tensors="pt",
                                      truncation=True,
                                      max_length=1024,
                                      padding=True)
            
                # Make prediction using BERT
                with torch.no_grad():
                    outputs = BERT_MODEL(**inputs)
                    probabilities = softmax(outputs.logits, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item()

                logger.info(f"Processing sentence: {sentence[:100]}...")
                logger.info(f"Prediction: {prediction}, Confidence: {confidence:.1%}")

                # If BERT flags this as non-compliant with sufficient confidence
                if prediction == 1 and len(sentence.split()) > 2:  # Non-compliant and meaningful length
                    try:
                        # Find a compliant alternative using a rule-based approach + example database
                        compliant_text, rule_violated, rationale = generate_compliant_alternative(
                            sentence, confidence, compliance_examples)
                        
                        # Only create instance if we have meaningful alternative
                        if compliant_text and len(compliant_text.strip()) > 5:
                            instance = {
                                "flagged_instance": sentence,
                                "compliance_status": "non-compliant",
                                "specific_compliant_alternative": compliant_text,
                                "rationale": f"{rule_violated}: {rationale}" if rule_violated else rationale,
                                "page": page_num,
                                "confidence": f"{confidence:.1%}"
                            }
                            flagged_instances.append(instance)
                            logger.info(f"Flagged instance on page {page_num}: {sentence}")
                            logger.info(f"Confidence: {confidence:.1%}")
                            logger.info(f"Compliant alternative: {compliant_text}")
                    
                    except Exception as e:
                        logger.error(f"Error generating alternative: {e}")
                        continue

            # Filter out any remaining invalid instances
            flagged_instances = [
                instance for instance in flagged_instances
                if instance.get("flagged_instance") 
                and len(instance["flagged_instance"].strip()) > 5
                and instance["flagged_instance"].strip() not in ['•', '\u2022']
                and float(instance["confidence"].rstrip('%')) > 70
            ]

            logger.info(f"Processed {len(sentences)} sentences, found {len(flagged_instances)} valid flagged instances")
            return {
                "compliant": len(flagged_instances) == 0,
                "flagged_instances": flagged_instances
            }
        
    except Exception as e:
        logger.error(f"Error during compliance check: {e}")
        return {"compliant": False, "error": "An error occurred during compliance checking."}


def generate_compliant_alternative(sentence, confidence, compliance_examples):
    """Generate a compliant alternative for a flagged sentence with enhanced rationale."""
    
    # First try to find a similar example in our database
    if compliance_examples:
        # Calculate similarity with each example
        best_match = None
        highest_score = 0
        
        for example in compliance_examples:
            if "non_compliant" in example and "compliant" in example:
                score = similarity_score(sentence, example["non_compliant"])
                if score > highest_score and score > 0.6:  # Reasonable similarity threshold
                    highest_score = score
                    best_match = example
        
        # If we found a good match, use its compliant version and "why" message
        if best_match:
            # Get the custom "why" message or use default if not available
            why_message = best_match.get("why", "Try using more balanced language and avoid promissory statements.")
            
            # Create the full rationale with standard prefix
            rationale = f"This text may not comply with FINRA regulations. {why_message}"
            
            return (
                best_match["compliant"], 
                "Rule 2", 
                rationale
            )
    
    # If no good match in examples, use rule-based alternative generation
    sentence_lower = sentence.lower()
    
    # Rule 1: Check for promissory language (will, guarantee, always, etc.)
    if any(word in sentence_lower for word in ["will", "guarantee", "guaranteed", "always", "never", "best"]):
        # Replace problematic words and phrases
        fixed_text = sentence
        fixed_text = re.sub(r'\bwill\b', 'may', fixed_text, flags=re.IGNORECASE)
        fixed_text = re.sub(r'\b(guarantee|guaranteed)\b', 'aim to provide', fixed_text, flags=re.IGNORECASE)
        fixed_text = re.sub(r'\balways\b', 'often', fixed_text, flags=re.IGNORECASE)
        fixed_text = re.sub(r'\bnever\b', 'rarely', fixed_text, flags=re.IGNORECASE)
        fixed_text = re.sub(r'\bbest\b', 'strong', fixed_text, flags=re.IGNORECASE)
        
        return (
            fixed_text,
            "Rule 2",
            "This text may not comply with FINRA regulations. Avoid using absolute or promissory language that suggests guaranteed outcomes."
        )
    
    # Rule 2: Check for performance projections
    elif any(term in sentence_lower for term in ["return", "performance", "gain", "profit", "grow", "increase"]):
        if "past performance" not in sentence_lower and "historical" not in sentence_lower:
            # Add risk disclaimer to performance statements
            risk_statement = " Past performance does not guarantee future results."
            if sentence.endswith(('.', '!', '?')):
                fixed_text = sentence[:-1] + risk_statement + sentence[-1]
            else:
                fixed_text = sentence + risk_statement
                
            return (
                fixed_text,
                "Rule 3",
                "This text may not comply with FINRA regulations. Statements regarding performance must include appropriate risk disclosures."
            )
    
    # Rule 3: Check for potential one-sided presentation (benefits without risks)
    elif any(term in sentence_lower for term in ["benefit", "advantage", "opportunity"]):
        if not any(term in sentence_lower for term in ["risk", "challenge", "consider", "however"]):
            balanced_statement = " It's important to consider both potential benefits and risks."
            if sentence.endswith(('.', '!', '?')):
                fixed_text = sentence[:-1] + balanced_statement + sentence[-1]
            else:
                fixed_text = sentence + balanced_statement
                
            return (
                fixed_text,
                "Rule 1",
                "This text may not comply with FINRA regulations. Communications must provide balanced treatment of risks and benefits."
            )
    
    # Default fallback if no specific rule applies
    fallback_text = sentence.replace("will", "may")
    return (
        fallback_text,
        "Rule 2",
        "This text may not comply with FINRA regulations. Try using more balanced language and avoid promissory statements."
    )

def load_compliance_examples():
    """Load examples of non-compliant text and their compliant alternatives."""
    try:
        with open('fcd.json', 'r') as f:
            examples = json.load(f)
            
            # Normalize format if needed
            if isinstance(examples, dict):
                examples = list(examples.values())
                
            # Filter to ensure we only have valid examples
            valid_examples = [
                example for example in examples
                if isinstance(example, dict) and "non_compliant" in example and "compliant" in example
            ]
            
            logger.info(f"Loaded {len(valid_examples)} compliance examples")
            return valid_examples
    except Exception as e:
        logger.error(f"Error loading compliance examples: {e}")
        return []


def similarity_score(text1, text2):
    """Calculate similarity between two text strings."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()



# APP = FLASK(__name__)
app = Flask(__name__)

@app.route('/intro')
def intro_page():
    return render_template('intro.html')

@app.route('/api/remove-user-completely', methods=['POST'])
def permanently_remove_user_from_system():
    try:
        # Get current user info based on how your app is structured
        user_email = session.get('user_email')
        
        if not user_email:
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        # Load all users to find current user
        with open('users.json', 'r') as f:
            users = json.load(f)
        
        # Find current user to check admin status
        current_user = None
        for user in users:
            if user.get('email') == user_email:
                current_user = user
                break
        
        # Check if user is admin
        if not current_user or current_user.get("Administrator Access NEW") != "Yes":
            return jsonify({"success": False, "error": "Unauthorized access"}), 403
        
        # Get the user email from the request
        data = request.json
        email_to_remove = data.get('email')
        
        if not email_to_remove:
            return jsonify({"success": False, "error": "Email is required"}), 400
        
        # Don't allow removing yourself
        if email_to_remove == user_email:
            return jsonify({"success": False, "error": "You cannot remove your own account"}), 400
        
        # Find and remove the user
        user_found = False
        for i, user in enumerate(users):
            if user.get('email') == email_to_remove:
                users.pop(i)
                user_found = True
                break
        
        if not user_found:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        # Save the updated user data
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=4)
        
        return jsonify({"success": True, "message": "User permanently removed from system"})
    
    except Exception as e:
        app.logger.error(f"Error removing user: {str(e)}")
        return jsonify({"success": False, "error": f"An error occurred: {str(e)}"}), 500
    
def get_user_data(email):
    """Retrieve user data from users.json file by email"""
    try:
        with open('users.json', 'r') as f:
            users_data = json.load(f)
            
        for user in users_data:
            if user.get('email') == email:
                return user
                
        return None  # User not found
    except Exception as e:
        print(f"Error reading user data: {str(e)}")
        return None

@app.route('/admin-access')
def admin_access_page():
    # Check if the user is logged in and is an admin
    user_email = session.get('user_email')
    if not user_email:
        return redirect('/login')
    
    # Get the user's data
    user_data = get_user_data(user_email)
    if not user_data or user_data.get("Administrator Access NEW") != "Yes":
        # Redirect non-admins to the profile page
        return redirect('/profile')
    
    return render_template('admin-access.html')

@app.route('/api/all-users', methods=['GET'])
def get_all_users():
    # Check if the user is logged in and is an admin
    user_email = session.get('user_email')
    if not user_email:
        return jsonify({'error': 'User not logged in'}), 401
    
    user_data = get_user_data(user_email)
    if not user_data or user_data.get("Administrator Access NEW") != "Yes":
        return jsonify({'error': 'Unauthorized access'}), 403
    
    # Get all users from the JSON file
    users = []
    try:
        with open('users.json', 'r') as f:
            users_data = json.load(f)
            # Return a simplified version with only necessary fields
            users = [
                {
                    'fullName': user.get('fullName', ''),
                    'email': user.get('email', ''),
                    'Administrator Access NEW': user.get('Administrator Access NEW', 'No')
                }
                for user in users_data
            ]
    except Exception as e:
        return jsonify({'error': f'Failed to load users: {str(e)}'}), 500
    
    return jsonify(users)

@app.route('/api/update-admin-access', methods=['POST'])
def update_admin_access():
    # Check if the user is logged in and is an admin
    user_email = session.get('user_email')
    if not user_email:
        return jsonify({'error': 'User not logged in'}), 401
    
    current_user = get_user_data(user_email)
    if not current_user or current_user.get("Administrator Access NEW") != "Yes":
        return jsonify({'error': 'Unauthorized access'}), 403
    
    # Get request data
    data = request.json
    target_email = data.get('email')
    admin_access = data.get('adminAccess')  # "Yes" or "No"
    
    if not target_email or admin_access not in ["Yes", "No"]:
        return jsonify({'error': 'Invalid request parameters'}), 400
    
    # Update the user's admin access in the users.json file
    try:
        with open('users.json', 'r') as f:
            users_data = json.load(f)
        
        # Find the user and update their admin access
        user_found = False
        for user in users_data:
            if user.get('email') == target_email:
                user["Administrator Access NEW"] = admin_access
                user_found = True
                break
        
        if not user_found:
            return jsonify({'error': 'User not found'}), 404
        
        # Save the updated users data
        with open('users.json', 'w') as f:
            json.dump(users_data, f, indent=4)
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': f'Failed to update admin access: {str(e)}'}), 500
    
# Store password reset tokens with expiration
password_reset_tokens = {}  # token -> {email, expiry}

@app.route('/reset-password', methods=['GET'])
def reset_password_page():
    return render_template('reset_password.html')

def generate_reset_token():
    """Generate a unique token for password reset"""
    return str(uuid.uuid4())

def send_password_reset_email(email, reset_link):
    """Send password reset email using SendGrid via the EmailService class"""
    return email_service.send_password_reset_email(email, reset_link)
    
@app.route('/reset-password', methods=['POST'])
def reset_password_request():
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    # Check if user exists
    users = load_users()
    user = next((u for u in users if u['email'] == email), None)
    
    if not user:
        # For security reasons, don't reveal that the email doesn't exist
        # Instead, pretend we sent an email
        return jsonify({'message': 'If your email exists in our system, you will receive a password reset link.'}), 200
    
    # Generate a token and set expiration time (24 hours from now)
    token = generate_reset_token()
    expiry = datetime.now() + timedelta(hours=24)
    
    # Store the token
    password_reset_tokens[token] = {
        'email': email,
        'expiry': expiry
    }
    
    # Create reset link
    reset_link = f"{request.host_url}reset?token={token}"
    
    # Send the reset email (simulation)
    success = send_password_reset_email(email, reset_link)
    
    if success:
        return jsonify({'message': 'Password reset link has been sent to your email.'}), 200
    else:
        return jsonify({'error': 'Failed to send reset email. Please try again later.'}), 500

@app.route('/reset', methods=['GET'])
def reset_password_form():
    token = request.args.get('token')
    
    if not token or token not in password_reset_tokens:
        return render_template('reset_error.html', error="Invalid or expired reset link"), 400
    
    token_data = password_reset_tokens[token]
    
    # Check if token is expired
    if datetime.now() > token_data['expiry']:
        del password_reset_tokens[token]  # Clean up expired token
        return render_template('reset_error.html', error="Reset link has expired"), 400
    
    return render_template('reset_password2.html', token=token)

@app.route('/reset', methods=['POST'])
def handle_password_reset():
    token = request.form.get('token')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not token or token not in password_reset_tokens:
        return render_template('reset_error.html', error="Invalid or expired reset link"), 400
    
    token_data = password_reset_tokens[token]
    
    # Check if token is expired
    if datetime.now() > token_data['expiry']:
        del password_reset_tokens[token]  # Clean up expired token
        return render_template('reset_error.html', error="Reset link has expired"), 400
    
    if not new_password or not confirm_password:
        return render_template('reset_password2.html', token=token, error="Please fill out all fields"), 400
    
    if new_password != confirm_password:
        return render_template('reset_password2.html', token=token, error="Passwords do not match"), 400
    
    # Update the user's password
    email = token_data['email']
    users = load_users()
    user = next((u for u in users if u['email'] == email), None)
    
    if user:
        user['newPassword'] = new_password  # Update password
        save_users(users)
        
        # Clean up the used token
        del password_reset_tokens[token]
        
        # Redirect to login page with success message
        return render_template('reset_password2.html', success=True)
    else:
        return render_template('reset_error.html', error="User not found"), 404    


@app.route('/process_saved_analysis', methods=['GET'])
def process_saved_analysis():
    # This will render a results page using data from the client
    # without needing the original file
    return render_template('results.html', 
                           load_from_client=True, 
                           finra_analysis=[])  # Empty placeholder

@app.route('/scan_file_for_disclosures', methods=['POST'])
def scan_file_for_disclosures():
    """Scan an uploaded file for keywords from disclosures.json and return matches by page"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        # Save the file temporarily
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)
        
        # Extract text by page from the file
        if file_path.endswith('.pdf'):
            text_by_page = extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text_by_page = extract_text_from_docx(file_path)
        else:
            os.remove(file_path)  # Clean up
            return jsonify({'success': False, 'error': 'Unsupported file type'}), 400
        
        # Load disclosures from disclosures.json
        DISCLOSURES_FILE = os.path.join(os.path.dirname(__file__), 'disclosures.json')
        disclosures = {}
        
        if os.path.exists(DISCLOSURES_FILE):
            try:
                with open(DISCLOSURES_FILE, 'r') as f:
                    disclosures_data = json.load(f)
                    
                    # Handle different formats of disclosures.json
                    if isinstance(disclosures_data, list) and len(disclosures_data) > 0:
                        disclosures = disclosures_data[0] if isinstance(disclosures_data[0], dict) else {}
                    else:
                        disclosures = disclosures_data
                
                logger.info(f"Loaded {len(disclosures)} disclosures for file scanning")
            except Exception as e:
                logger.error(f"Error loading disclosures.json: {e}")
                # Fallback disclosures if loading fails
                disclosures = {
                    "mutual fund": "Mutual fund investing involves risk; principal loss is possible.",
                    "index fund": "Index funds are subject to market risk, including the possible loss of principal.",
                    "performance": "Past performance does not guarantee future results.",
                    "investing": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                    "investment": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                    "market": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                    "funds": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                    "returns": "Past performance is no guarantee of future results."
                }
        else:
            logger.info(f"Disclosures file not found for scanning")
            disclosures = {
                "mutual fund": "Mutual fund investing involves risk; principal loss is possible.",
                "index fund": "Index funds are subject to market risk, including the possible loss of principal.",
                "performance": "Past performance does not guarantee future results.",
                "investing": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                "investment": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                "market": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                "funds": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                "returns": "Past performance is no guarantee of future results."
            }
        
        # Map of page numbers to their specific disclosures
        page_specific_disclosures = {}
        
        # Process each page for specific disclosures
        for page_data in text_by_page:
            page_num = page_data.get('page', 0)
            page_text = page_data.get('text', '')
            
            if not page_text:
                continue
            
            logger.info(f"Scanning page {page_num} for disclosure keywords")
            
            # Convert to lowercase for case-insensitive matching
            page_text_lower = page_text.lower()
            page_matches = []
            
            # Check each keyword for a match in the page text
            for keyword, description in disclosures.items():
                keyword_lower = keyword.lower()
                
                # If the keyword is found in the page text
                if keyword_lower in page_text_lower:
                    # Add the match with both keyword and description
                    page_matches.append({
                        "keyword": keyword,
                        "description": description
                    })
                    logger.info(f"Page {page_num}: Matched keyword '{keyword}'")
            
            # Only add pages that have matches
            if page_matches:
                page_specific_disclosures[str(page_num)] = page_matches
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'page_specific_disclosures': page_specific_disclosures
        })
        
    except Exception as e:
        logger.error(f"Error scanning file for disclosures: {e}")
        # Try to clean up the file if it exists
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/training')
def training():
    return render_template('training.html')

# Add this new route to check if advisor type is selected
@app.route('/api/advisor-type-status', methods=['GET'])
def advisor_type_status():
    if 'user_email' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)
    
    if user:
        has_advisor_type = 'advisorType' in user and user['advisorType']
        return jsonify({
            'hasAdvisorType': has_advisor_type,
            'signUpStatus': user.get('Sign Up Status', 'Not Yet')
        })
    
    return jsonify({'error': 'User not found'}), 404

@app.route('/get_page_specific_disclosures', methods=['POST'])
def get_page_specific_disclosures():
    """Get page-specific disclosures based on exact keyword matching for each page"""
    data = request.get_json()
    content_type = data.get('content_type', '')  # 'Book' or 'Presentation'
    text = data.get('text', '')  # Full document text
    original_text_by_page = data.get('original_text_by_page', [])  # Text split by pages
    
    logger.info(f"Processing page-specific disclosures for {content_type}")
    logger.info(f"Received {len(original_text_by_page)} pages to analyze")
    
    # Debug log the specific pages we received
    for i, page in enumerate(original_text_by_page):
        page_num = page.get('page', i+1)
        page_text = page.get('text', '')
        logger.info(f"Page {page_num}: {page_text[:100]}...")
    
    if not content_type or (not text and not original_text_by_page):
        logger.info(f"Missing required params: content_type={content_type}, pages={len(original_text_by_page)}")
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Load disclosures from disclosures.json
        DISCLOSURES_FILE = os.path.join(os.path.dirname(__file__), 'disclosures.json')
        disclosures = {}
        
        if os.path.exists(DISCLOSURES_FILE):
            try:
                with open(DISCLOSURES_FILE, 'r') as f:
                    disclosures_data = json.load(f)
                    
                    # Handle different formats of disclosures.json
                    if isinstance(disclosures_data, list) and len(disclosures_data) > 0:
                        disclosures = disclosures_data[0] if isinstance(disclosures_data[0], dict) else {}
                    else:
                        disclosures = disclosures_data
                
                logger.info(f"Loaded {len(disclosures)} disclosures from {DISCLOSURES_FILE}")
                
                # Log a sample of the keywords to help debug
                sample_keys = list(disclosures.keys())[:5]
                logger.info(f"Sample keywords: {sample_keys}")
            except Exception as e:
                logger.error(f"Error loading disclosures.json: {e}")
                # Add fallback disclosures if loading fails
                disclosures = {
                    "mutual fund": "Mutual fund investing involves risk; principal loss is possible.",
                    "index fund": "Index funds are subject to market risk, including the possible loss of principal.",
                    "performance": "Past performance does not guarantee future results.",
                    "investing": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                    "investment": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                    "market": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                    "funds": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                    "returns": "Past performance is no guarantee of future results."
                }
        else:
            logger.info(f"Disclosures file not found: {DISCLOSURES_FILE}")
            # Add fallback disclosures if file doesn't exist
            disclosures = {
                "mutual fund": "Mutual fund investing involves risk; principal loss is possible.",
                "index fund": "Index funds are subject to market risk, including the possible loss of principal.",
                "performance": "Past performance does not guarantee future results.",
                "investing": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                "investment": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                "market": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                "funds": "Investing involves risk including loss of principal. No strategy assures success or protects against loss.",
                "returns": "Past performance is no guarantee of future results."
            }
        
        # Log all keywords for debugging
        logger.info(f"Available disclosure keywords: {list(disclosures.keys())}")
        
        # Map of page numbers to their specific disclosures
        page_disclosures = {}
        matched_keywords = {}  # Track which keywords matched on which pages
        
        # Process each page for specific disclosures
        for page_data in original_text_by_page:
            page_num = page_data.get('page', 0)
            page_text = page_data.get('text', '')
            
            if not page_text:
                logger.info(f"Skipping page {page_num}: No text content")
                continue
            
            logger.info(f"Processing page {page_num} with {len(page_text)} characters")
            logger.info(f"Page {page_num} sample: {page_text[:100]}...")
            
            # Convert to lowercase for case-insensitive matching
            page_text_lower = page_text.lower()
            current_page_disclosures = []
            matched_keywords[page_num] = []
            
            # Check each keyword for an exact match in the page text
            for keyword, disclosure in disclosures.items():
                keyword_lower = keyword.lower()
                
                # If the keyword is found in the page text
                if keyword_lower in page_text_lower:
                    # Add the disclosure if not already present
                    if disclosure not in current_page_disclosures:
                        current_page_disclosures.append(disclosure)
                        matched_keywords[page_num].append(keyword)
                        logger.info(f"Page {page_num}: Matched keyword '{keyword}'")
            
            # Special handling for investing-related terms
            investing_terms = ["invest", "investing", "investment", "mutual fund", "funds", "return", "returns"]
            has_investing_term = any(term in page_text_lower for term in investing_terms)
            
            if has_investing_term:
                investing_disclosure = "Investing involves risk including loss of principal. No strategy assures success or protects against loss."
                if investing_disclosure not in current_page_disclosures:
                    current_page_disclosures.append(investing_disclosure)
                    matched_keywords[page_num].append("investment terms")
                    logger.info(f"Page {page_num}: Added investing disclosure based on investment terms")
            
            # Always add standard disclosures for test pages to demonstrate functionality
            if "mutual fund" in page_text_lower:
                mutual_fund_disclosure = "Mutual fund investing involves risk; principal loss is possible."
                if mutual_fund_disclosure not in current_page_disclosures:
                    current_page_disclosures.append(mutual_fund_disclosure)
                    matched_keywords[page_num].append("mutual fund")
            
            if "index fund" in page_text_lower:
                index_fund_disclosure = "Index funds are subject to market risk, including the possible loss of principal."
                if index_fund_disclosure not in current_page_disclosures:
                    current_page_disclosures.append(index_fund_disclosure)
                    matched_keywords[page_num].append("index fund")
            
            if "cd" in page_text_lower.split():  # Match "CD" as a whole word
                cd_disclosure = "CDs are FDIC insured up to applicable limits and offer a fixed rate of return."
                if cd_disclosure not in current_page_disclosures:
                    current_page_disclosures.append(cd_disclosure)
                    matched_keywords[page_num].append("CD")
            
            # Only add pages that have disclosures
            if current_page_disclosures:
                # Store page number as string for JSON compatibility
                page_disclosures[str(page_num)] = current_page_disclosures
                logger.info(f"Page {page_num}: Added {len(current_page_disclosures)} disclosures")
        
        # If no page disclosures were found, return an empty object
        if not page_disclosures:
            logger.info("No page-specific disclosures found")
        else:
            logger.info(f"Found page-specific disclosures for pages: {list(page_disclosures.keys())}")
        
        # Add a general disclosure to ensure there's always something to display
        general_disclosures = ["Content in this material is for general information only and not intended to provide specific advice or recommendations for any individual."]
        
        return jsonify({
            'page_disclosures': page_disclosures,
            'general_disclosures': general_disclosures,
            'debug_info': {
                'matched_keywords': matched_keywords,
                'keywords_available': list(disclosures.keys())
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating page-specific disclosures: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_file_pages', methods=['POST'])
def get_file_pages():
    """Get page-by-page text for a processed file"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'success': False, 'error': 'Filename is required'}), 400
    
    try:
        # Check if the file has been processed
        if filename in processed_files and 'text_by_page' in processed_files[filename]:
            text_by_page = processed_files[filename]['text_by_page']
            
            # Log what we're returning
            logger.info(f"Returning {len(text_by_page)} pages for file: {filename}")
            
            return jsonify({
                'success': True,
                'text_by_page': text_by_page
            })
        else:
            logger.info(f"File {filename} not found in processed_files or missing text_by_page")
            return jsonify({'success': False, 'error': 'File not found or processing incomplete'}), 404
    
    except Exception as e:
        logger.error(f"Error retrieving file pages: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/get_user_scenario', methods=['GET'])
def get_user_scenario():
    """Get the assigned scenario for the current user."""
    try:
        # Get the current user from session
        if 'user_email' not in session:
            return jsonify({"success": False, "error": "User not logged in"}), 401
        
        user_email = session['user_email']
        
        # Load users from JSON
        users = load_users()
        
        # Find the user
        user = next((u for u in users if u['email'] == user_email), None)
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        # Check if user has an assigned scenario
        if 'assignedScenarios' in user and user['assignedScenarios'] and len(user['assignedScenarios']) > 0:
            scenario_title = user['assignedScenarios'][0]
            
            # Load scenarios to get the full text
            scenarios = {}
            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, 'r') as file:
                    scenarios = json.load(file)
            
            # Get the scenario text
            scenario_text = scenarios.get(scenario_title, scenario_title)
            
            return jsonify({
                "success": True, 
                "hasScenario": True,
                "scenarioTitle": scenario_title,
                "scenarioText": scenario_text
            })
        
        # No scenario assigned
        return jsonify({"success": True, "hasScenario": False})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# Remove scenario from user
@app.route('/remove_user_scenario', methods=['POST'])
def remove_user_scenario():
    """Remove assigned scenario from a user."""
    try:
        data = request.get_json()
        user_email = data.get('email')
        
        # Log the incoming request for debugging
        print(f"Removal request for email: {user_email}")
        
        if not user_email:
            return jsonify({"success": False, "error": "User email is required"}), 400

        # Load users from the JSON file
        users = load_users()

        # Find the user by email
        user = next((u for u in users if u['email'] == user_email), None)
        if user:
            # Remove the assigned scenarios
            if 'assignedScenarios' in user:
                user['assignedScenarios'] = []
            
            # Also clear the advisor type
            user['advisorType'] = ''
            
            # Save the updated user data
            save_users(users)
            return jsonify({"success": True, "message": f"Scenario removed from user '{user_email}'."}), 200
        else:
            return jsonify({"success": False, "error": f"User '{user_email}' not found."}), 404
    except Exception as e:
        print(f"Error in remove_user_scenario: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/assign-disclosures')
def assign_disclosures_page():
    return render_template('assign_disclosures.html')

@app.route("/get_relevant_disclosures", methods=["POST"])
def get_relevant_disclosures():
    data = request.get_json()
    revised_text = data.get("revised_text", "")
    
    logger.info(f"Generating disclosures for revised text: '{revised_text[:50]}...'")
    
    # Get the relevant disclosures based on keywords in the revised text
    matched_disclosures = get_matching_disclosures(revised_text)
    
    # Format the disclosures into a single paragraph
    disclosure_paragraph = " ".join(matched_disclosures)
    
    return jsonify({
        "disclosure_paragraph": disclosure_paragraph
    })

@app.route('/get_disclosures', methods=['GET'])
def get_disclosures():
    """Get standard disclosures from disclosures.json"""
    try:
        # File path to disclosures.json
        DISCLOSURES_FILE = os.path.join(os.path.dirname(__file__), 'disclosures.json')
        
        if os.path.exists(DISCLOSURES_FILE):
            with open(DISCLOSURES_FILE, 'r') as f:
                disclosures_data = json.load(f)
            
            # Convert the data to a regular dictionary for the frontend
            if isinstance(disclosures_data, list) and len(disclosures_data) > 0:
                # In this case, it's a list with a single object containing all disclosures
                disclosures = disclosures_data[0] if isinstance(disclosures_data[0], dict) else {}
            else:
                # If it's already a dictionary, use it as is
                disclosures = disclosures_data
                
            return jsonify({"success": True, "disclosures": disclosures}), 200
        else:
            return jsonify({"success": False, "error": "Disclosures file not found", "disclosures": {}}), 404
    except Exception as e:
        logger.error(f"Error reading disclosures.json: {e}")
        return jsonify({"success": False, "error": str(e), "disclosures": {}}), 500

@app.route('/update_disclosure', methods=['POST'])
def update_disclosure():
    """Update disclosures in disclosures.json"""
    try:
        # File path to disclosures.json
        DISCLOSURES_FILE = os.path.join(os.path.dirname(__file__), 'disclosures.json')
        
        # Get request data
        data = request.json
        action = data.get('action')
        topic = data.get('topic')
        
        if not topic or not action:
            return jsonify({
                "success": False, 
                "error": "Missing required parameters"
            }), 400
        
        # Load existing disclosures
        if os.path.exists(DISCLOSURES_FILE):
            with open(DISCLOSURES_FILE, 'r') as f:
                disclosures_data = json.load(f)
        else:
            # Create a new structure that matches what we observed
            disclosures_data = [{}]
        
        # Handle the specific JSON structure (list with single object)
        if isinstance(disclosures_data, list) and len(disclosures_data) > 0:
            # Work with the first item in the list
            disclosures = disclosures_data[0]
            
            if action == 'add':
                disclosure_text = data.get('disclosure')
                if not disclosure_text:
                    return jsonify({
                        "success": False, 
                        "error": "Missing disclosure text"
                    }), 400
                
                # Add or update the disclosure
                disclosures[topic] = disclosure_text
                
            elif action == 'delete':
                # Remove the disclosure if it exists
                if topic in disclosures:
                    del disclosures[topic]
                else:
                    return jsonify({
                        "success": False, 
                        "error": f"Disclosure topic '{topic}' not found"
                    }), 404
            else:
                return jsonify({
                    "success": False, 
                    "error": f"Invalid action: {action}"
                }), 400
        else:
            # If the structure is different, initialize it
            disclosures_data = [{}]
            if action == 'add':
                disclosure_text = data.get('disclosure')
                if not disclosure_text:
                    return jsonify({
                        "success": False, 
                        "error": "Missing disclosure text"
                    }), 400
                
                # Add the disclosure to the new structure
                disclosures_data[0][topic] = disclosure_text
            else:
                return jsonify({
                    "success": False, 
                    "error": "No disclosures to delete"
                }), 404
        
        # Save the updated disclosures
        with open(DISCLOSURES_FILE, 'w') as f:
            json.dump(disclosures_data, f, indent=4)
        
        return jsonify({"success": True}), 200
        
    except Exception as e:
        logger.error(f"Error updating disclosures.json: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# First, keep the endpoint for API calls
@app.route('/get_matching_disclosures', methods=['POST'])
def get_matching_disclosures_api():
    """API endpoint to get matching disclosures for given text"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
        
    matched_disclosures = get_matching_disclosures(text)
    return jsonify({'disclosures': matched_disclosures})

# Then, define the helper function separately
def get_matching_disclosures(text):
    """
    Analyze text for keywords from disclosures.json and return only matching disclosures.
    """
    # Load disclosures from disclosures.json
    DISCLOSURES_FILE = os.path.join(os.path.dirname(__file__), 'disclosures.json')
    disclosures = {}
    
    try:
        if os.path.exists(DISCLOSURES_FILE):
            with open(DISCLOSURES_FILE, 'r') as f:
                disclosures_data = json.load(f)
                
                # Handle different formats of disclosures.json
                if isinstance(disclosures_data, list) and len(disclosures_data) > 0:
                    disclosures = disclosures_data[0] if isinstance(disclosures_data[0], dict) else {}
                else:
                    disclosures = disclosures_data
                    
            logger.info(f"Loaded {len(disclosures)} disclosures from {DISCLOSURES_FILE}")
    except Exception as e:
        logger.error(f"Error loading disclosures: {e}")
        return ["Error loading disclosures."]
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # List to store matched disclosures
    matched_disclosures = []
    matched_keywords = []
    
    # Add standard general disclosure
    general_disclosure = "Content in this material is for general information only and not intended to provide specific advice or recommendations for any individual."
    matched_disclosures.append(general_disclosure)
    
    # Look for specific keywords in the text
    for keyword, disclosure in disclosures.items():
        keyword_lower = keyword.lower()
        
        # Check if the keyword is in the text
        if keyword_lower in text_lower:
            matched_keywords.append(keyword)
            
            # Only add the disclosure if we haven't added it yet
            if disclosure not in matched_disclosures:
                matched_disclosures.append(disclosure)
                logger.info(f"Added disclosure for keyword: {keyword}")
    
    # Always include investing disclosure if investing-related terms are found
    investing_terms = ["invest", "investing", "investment", "mutual fund", "funds"]
    if any(term in text_lower for term in investing_terms):
        investing_disclosure = "Investing involves risk including loss of principal. No strategy assures success or protects against loss."
        if investing_disclosure not in matched_disclosures:
            matched_disclosures.append(investing_disclosure)
    
    logger.info(f"Found {len(matched_keywords)} matching keywords: {matched_keywords}")
    logger.info(f"Returning {len(matched_disclosures)} matched disclosures")
    
    return matched_disclosures

@app.route('/manage-disclosures')
def manage_disclosures():
    return render_template('manage_disclosures.html')


def load_training_data(json_file):
    """Load and preprocess training data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    for item in data:
        texts.append(item['non_compliant'])
        texts.append(item['compliant']) 
        labels.append(1)  # Non-compliant
        labels.append(0)  # Compliant
        
    return texts, labels

def save_model(model, tokenizer, model_path='finra_compliance_model.pth', tokenizer_path='finra_tokenizer'):
    """Save the trained model and tokenizer."""
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)


# BERT Functionality

def predict_compliance_bert(text, model, tokenizer):
    # Prepare input
    inputs = tokenizer(text, 
                      return_tensors="pt",
                      truncation=True,
                      max_length=512,
                      padding=True)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence








# Function for checking subscription status in Profile HTML
@app.route('/api/subscription-status', methods=['GET'])
def subscription_status():
    email = request.args.get('email')
    if not email:
        return jsonify({'error': 'Email parameter is required'}), 400
    users = load_users()
    user = next((u for u in users if u['email'] == email), None)
    if user:
        return jsonify({'subscribed': user.get('Subscribed') == "Yes"})
    return jsonify({'error': 'User not found'}), 404

# Example Flask route to increment login count
@app.route('/track_login', methods=['POST'])
def track_login():
    user_email = request.json.get('email')
    if not user_email:
        return jsonify({'error': 'Email required'}), 400

    # Fetch user from the database
    user = db.session.query(User).filter_by(email=user_email).first()
    if user:
        user.login_count = (user.login_count or 0) + 1
        db.session.commit()
        return jsonify({'login_count': user.login_count})
    else:
        return jsonify({'error': 'User not found'}), 404


# Load users.json data
def loads_users():
    with open("users.json", "r") as file:
        return json.load(file)

# Identifying Current User in Profile HTML for updating the Advisor Type
@app.route('/api/current-user', methods=['GET'])
def get_current_user():
    if 'user_email' in session:
        user_email = session['user_email']
        logger.info(f"Session user_email: {user_email}")  # Debugging
        users = load_users()
        user = next((u for u in users if u['email'] == user_email), None)
        if user:
            
            # Ensure `usage` and `usage_denominator` are present
            user['usage'] = user.get('usage', 0.0001)
            user['usage_denominator'] = user.get('usage_denominator', 100)  # Default denominator

            # Safeguard: Only modify AccessEndDate if missing or invalid
            if not user.get('AccessEndDate') or user['AccessEndDate'] == '':
                logger.info(f"AccessEndDate is missing or invalid for {user_email}. Setting default value.")
                user['AccessEndDate'] = calculate_next_billing_date()  # Update only if necessary
            
            # Optional: Log AccessEndDate for debugging
            logger.info(f"AccessEndDate for {user_email}: {user['AccessEndDate']}")

            # No changes needed here - the entire user object is already being returned
            # If advisorType exists in the user data, it will be included automatically
            
            return jsonify(user)
    return jsonify({'error': 'User not found or not logged in'}), 401


@app.route('/update-profile', methods=['POST'])
def update_profile():
    data = request.json
    if 'user_email' not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    
    current_email = session['user_email']
    users = load_users()
    
    # Find the user to update
    for user in users:
        if user.get('email') == current_email:
            # Update basic profile information
            first_name = data.get('firstName', '')
            last_name = data.get('lastName', '')
            user['fullName'] = f"{first_name} {last_name}"
            
            # Update advisor type if provided
            if 'advisorType' in data:
                user['advisorType'] = data['advisorType']
            
            # Save users back to file
            save_users(users)
            logger.info(f"Updated profile for {current_email}, including advisorType: {data.get('advisorType', 'not provided')}")
            
            return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "User not found"}), 404

# Helper function to save users (assuming you have a similar load_users function)
def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=4)



# ASSIGNING USERS SCENARIOS for disclosure

# Endpoint to delete a scenario
@app.route('/delete_scenario', methods=['POST'])
def delete_scenario():
    """Delete a disclosure scenario."""
    try:
        data = request.json
        title = data.get('title')
        if not title:
            return jsonify({"success": False, "error": "Scenario title is required."}), 400

        # Load existing scenarios
        if os.path.exists(JSON_FILE_PATH):
            with open(JSON_FILE_PATH, 'r') as file:
                scenarios = json.load(file)
                
                # Ensure the file contains a dictionary
                if not isinstance(scenarios, dict):
                    raise ValueError("JSON file does not contain a valid dictionary.")
        else:
            return jsonify({"success": False, "error": "No scenarios found."}), 404

        # Check if the title exists in the scenarios
        if title not in scenarios:
            return jsonify({"success": False, "error": "Scenario not found."}), 404

        # Remove the scenario
        del scenarios[title]

        # Save the updated scenarios back to the file
        with open(JSON_FILE_PATH, 'w') as file:
            json.dump(scenarios, file, indent=4)
        return jsonify({"success": True, "message": f"Scenario '{title}' deleted successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Path to the JSON file
JSON_FILE_PATH = "scenarios.json"

@app.route('/save_advisor_type', methods=['POST'])
def save_advisor_type():
    try:
        data = request.json
        email = data.get('email')
        advisor_type = data.get('advisorType')
        
        if not email:
            return jsonify({"success": False, "error": "Email is required"}), 400
        
        # Load users from the JSON file
        users = load_users()
        
        # Find the user by email
        user_found = False
        for user in users:
            if user['email'] == email:
                user['advisorType'] = advisor_type  # Add/update advisorType
                user_found = True
                break
        
        if not user_found:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        # Save users back to file
        save_users(users)
        
        # Now update the user's assigned scenario to match the advisor type
        if advisor_type:
            # Check if this advisor type exists in scenarios.json
            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, 'r') as file:
                    scenarios = json.load(file)
                    
                    # Check if this advisor type exists as a key in the scenarios dictionary
                    if advisor_type in scenarios:
                        # Update the user's assigned scenario
                        result = assign_scenario_to_user(email, advisor_type)
                        if not result.get('success'):
                            logger.error(f"Error assigning scenario: {result.get('error')}")
                    else:
                        logger.info(f"Advisor type '{advisor_type}' not found in scenarios.json")
            else:
                logger.info("scenarios.json not found")
        elif advisor_type == '':
            # Clear the user's assigned scenario if advisor type is empty
            remove_scenario_from_user(email)
        
        return jsonify({"success": True, "message": "Advisor type saved successfully"})
    except Exception as e:
        logger.error(f"Error saving advisor type: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
def assign_scenario_to_user(email, scenario_title):
    """Assign a scenario to a user by updating their user record."""
    try:
        users = load_users()
        user_found = False
        
        for user in users:
            if user.get('email') == email:
                if 'assignedScenarios' not in user:
                    user['assignedScenarios'] = []
                user['assignedScenarios'] = [scenario_title]
                user_found = True
                break
                
        if not user_found:
            return {'success': False, 'error': 'User not found'}
            
        save_users(users)
        return {'success': True}
    except Exception as e:
        logger.error(f"Error assigning scenario to user: {e}")
        return {'success': False, 'error': str(e)}

def remove_scenario_from_user(email):
    """Remove assigned scenario from a user by updating their user record."""
    try:
        users = load_users()
        user_found = False
        
        for user in users:
            if user.get('email') == email:
                user['assignedScenarios'] = []
                user_found = True
                break
                
        if not user_found:
            return {'success': False, 'error': 'User not found'}
            
        save_users(users)
        return {'success': True}
    except Exception as e:
        logger.error(f"Error removing scenario from user: {e}")
        return {'success': False, 'error': str(e)}

@app.route('/update_advisor_type', methods=['POST'])
def update_advisor_type():
    try:
        data = request.json
        email = data.get('email')
        advisor_type = data.get('advisorType')
        
        if not email:
            return jsonify({"success": False, "error": "Email is required"}), 400
        
        # Load users from the JSON file
        users = load_users()
        
        # Find the user by email
        user_found = False
        for user in users:
            if user['email'] == email:
                user['advisorType'] = advisor_type  # Add/update advisorType
                user_found = True
                break
        
        if not user_found:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        # Save users back to file
        save_users(users)
        
        return jsonify({"success": True, "message": "Advisor type updated successfully"})
    except Exception as e:
        logger.error(f"Error updating advisor type: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
# Endpoint to update user scenario
@app.route('/update_user_scenario', methods=['POST'])
def update_user_scenario():
    try:
        data = request.json
        user = data.get('user')
        scenario = data.get('scenario')
        if not user or not scenario:
            return jsonify({"success": False, "error": "User or scenario missing."}), 400

        # Load existing data
        if os.path.exists(JSON_FILE_PATH):
            with open(JSON_FILE_PATH, 'r') as file:
                try:
                    scenarios = json.load(file)
                    
                    # Ensure the file contains a dictionary
                    if not isinstance(scenarios, dict):
                        raise ValueError("JSON file does not contain a valid dictionary.")
                except (json.JSONDecodeError, ValueError) as e:

                    # If the JSON is invalid or not a dictionary, reset it
                    scenarios = {}
        else:
            scenarios = {}

        # Update the JSON file with the new scenario
        scenarios[user] = scenario.get('disclosure', '')
        with open(JSON_FILE_PATH, 'w') as file:
            json.dump(scenarios, file, indent=4)
        return jsonify({"success": True, "message": "Scenario successfully updated."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Endpoint to add a new scenario
@app.route('/add_scenario', methods=['POST'])
def add_scenario():
    """Add a new disclosure scenario."""
    try:
        data = request.json
        title = data.get('title')
        disclosure = data.get('disclosure')

        if not title or not disclosure:
            return jsonify({"success": False, "error": "Scenario title or disclosure is missing."}), 400

        # Load existing scenarios
        if os.path.exists(JSON_FILE_PATH):
            with open(JSON_FILE_PATH, 'r') as file:
                try:
                    scenarios = json.load(file)

                    # Ensure the file contains a dictionary
                    if not isinstance(scenarios, dict):
                        raise ValueError("JSON file does not contain a valid dictionary.")
                except (json.JSONDecodeError, ValueError):
                    scenarios = {}
        else:
            scenarios = {}

        # Add the new scenario
        if title in scenarios:
            return jsonify({"success": False, "error": "Scenario title already exists."}), 400
        scenarios[title] = disclosure

        # Save the updated scenarios back to the file
        with open(JSON_FILE_PATH, 'w') as file:
            json.dump(scenarios, file, indent=4)
        return jsonify({"success": True, "message": "Scenario added successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Endpoint to fetch all scenarios
@app.route('/get_scenarios', methods=['GET'])
def get_scenarios():
    """Get all disclosure scenarios."""
    try:
        # Check if the JSON file exists
        if os.path.exists(JSON_FILE_PATH):
            with open(JSON_FILE_PATH, 'r') as file:
                scenarios = json.load(file)
                # Ensure the file contains a dictionary
                if not isinstance(scenarios, dict):
                    raise ValueError("JSON file does not contain a valid dictionary.")
        else:
            scenarios = {}

        # Convert scenarios into a list of titles and disclosures for the response
        scenario_list = [{"title": title, "disclosure": disclosure} for title, disclosure in scenarios.items()]
        return jsonify({"success": True, "scenarios": scenario_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Load assignments from JSON
def load_assignments():
    with open('assignments.json', 'r') as f:
        return json.load(f)

@app.route('/get_assigned_disclosure', methods=['POST'])
def get_assigned_disclosure():
    try:
        data = request.json
        user = data.get('user')  # Get the logged-in user from the request

        # Load assignments from the JSON file
        assignments = load_assignments()

        # Check if the user has an assigned scenario
        if user in assignments:
            return jsonify(success=True, disclosure=assignments[user])
        else:
            return jsonify(success=True, disclosure=None)  # No scenario assigned
    except Exception as e:
        return jsonify(success=False, error=str(e))


SCENARIOS_FILE = 'scenarios.json'


# Load scenarios from a file
def load_scenarios():
    try:
        with open(SCENARIOS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save scenarios to a file
def save_scenarios(scenarios):
    with open(SCENARIOS_FILE, 'w') as f:
        json.dump(scenarios, f, indent=4)


# Assigns Scenario
@app.route('/assign_scenario', methods=['POST'])
def assign_scenario():
    """Assign a scenario to a user."""
    try:
        data = request.get_json()
        user_email = data.get('email') or data.get('user')  # Accept either 'email' or 'user' parameter
        scenario_title = data.get('scenario')
        if not user_email or not scenario_title:
            return jsonify({"success": False, "error": "User and scenario are required"}), 400

        # Load users from the JSON file
        users = load_users()

        # Find the user by email
        user = next((u for u in users if u['email'] == user_email), None)
        if user:
            
            # Add the scenario to the user's assigned scenarios
            if 'assignedScenarios' not in user:
                user['assignedScenarios'] = []
            
            # Replace existing scenarios with the new one
            user['assignedScenarios'] = [scenario_title]
            
            # Also update the user's advisor type to match the assigned scenario
            user['advisorType'] = scenario_title
            
            # Save the updated user data
            save_users(users)
            return jsonify({"success": True, "message": f"Scenario '{scenario_title}' assigned to user '{user_email}'."}), 200
        else:
            return jsonify({"success": False, "error": f"User '{user_email}' not found."}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500    

#Search for existing user
@app.route('/search_user', methods=['POST'])
def search_user():
    """Search for a user account by username or email."""
    data = request.get_json()
    query = data.get('query', '').strip().lower()
    if not query:
        return jsonify({"success": False, "message": "Search query is required"}), 400
    
    # Load users from the JSON file
    users = load_users()  # Ensure this function loads the `users.json` file correctly

    # Find matching users
    matching_users = [
        user for user in users
        if query in user.get('email', '').strip().lower() or query in user.get('fullName', '').strip().lower()
    ]

    if matching_users:
        return jsonify({"success": True, "users": matching_users}), 200
    else:
        return jsonify({"success": False, "message": "No matching users found."}), 404





# CUSTOM DISCLOSURES (may be redundant)


# Function to create custom disclosures
@app.route('/get_custom_disclosures', methods=['GET'])
def get_custom_disclosures():
    
    # Load custom disclosures from the JSON file
    try:
        with open(CUSTOM_DISCLOSURES_FILE, 'r') as f:
            custom_disclosures = json.load(f)
        return jsonify({"disclosures": custom_disclosures}), 200
    except FileNotFoundError:
        return jsonify({"disclosures": {}}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to delete Custom Disclosures 
@app.route('/delete_custom_disclosure', methods=['POST'])
def delete_custom_disclosure():
    data = request.get_json()
    word_to_delete = data.get('word', '').strip()
    if not word_to_delete:
        return jsonify({"success": False, "error": "No word specified"}), 400
    try:
        with open(CUSTOM_DISCLOSURES_FILE, 'r') as f:
            custom_disclosures = json.load(f)
        if word_to_delete in custom_disclosures:
            del custom_disclosures[word_to_delete]

            # Save the updated disclosures back to the file
            with open(CUSTOM_DISCLOSURES_FILE, 'w') as f:
                json.dump(custom_disclosures, f, indent=4)
                
            # Also update in-memory DISCLOSURE_WORDS dictionary
            if word_to_delete in DISCLOSURE_WORDS:
                del DISCLOSURE_WORDS[word_to_delete]
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "error": "Word not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




# Set the secret key for sessions
app.secret_key = 's3cr3t_k3y_@1234'

# File path for users.json
USER_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

# Ensure users.json exists
if not os.path.exists(USER_FILE):
    with open(USER_FILE, 'w') as file:
        json.dump([], file)  # Initialize with an empty list
    logger.info(f"Created {USER_FILE} with an empty list.")



# Updates Users Usage

def update_user_usage(user_email, sentence_count=None, usage_increment=0.0):

    """Updates the user's usage based on sentence count or a fixed usage increment."""

    users = load_users()

    user = next((u for u in users if u['email'] == user_email), None)



    if user:

        if sentence_count is not None:

            cost_per_sentence = 0.0175

            additional_cost = sentence_count * cost_per_sentence

            user['usage'] += additional_cost

            logger.info(f"Updated usage for {user_email}: +${additional_cost:.2f}, Total: ${user['usage']:.2f}")



        if usage_increment > 0:

            user['usage'] += usage_increment

            logger.info(f"Updated usage for {user_email}: +${usage_increment:.2f}, Total: ${user['usage']:.2f}")



        save_users(users)

    else:

        logger.info(f"User with email {user_email} not found.")





# BACKEND HTML FUNCTIONALITY [Users Control (Add/Delete/Assign Admin)]
USERS_FILE = 'users.json'


# API to fetch all users BACKEND HTML
@app.route('/api/users', methods=['GET'])
def get_users():
    
    try:
        with open(USERS_FILE, 'r') as file:
            users = json.load(file)
        return jsonify(users), 200
    except FileNotFoundError:
        return jsonify({"error": "Users file not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding JSON file"}), 500

# Render the backend.html page on browser BACKEND HTML
@app.route('/users')
def users_list_page():
    return render_template('backend.html')  # Replace 'users_list.html' with the filename

# Delete a user in the BACKEND HTML
@app.route('/api/delete-user', methods=['POST'])
def delete_user():
    data = request.json
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    users = load_users()
    updated_users = [user for user in users if user['email'] != email]
    if len(users) == len(updated_users):
        return jsonify({'error': 'User not found'}), 404
    save_users(updated_users)
    return jsonify({'message': f'User with email {email} deleted successfully'}), 200


# Function to read the Users.json for Admin "Yes" or Admin "No" BACKEND HTML

@app.route('/api/add-admin', methods=['POST'])
def add_admin():
    data = request.json
    email = data.get('email')
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    try:
        # Load users
        with open('users.json', 'r') as f:
            users = json.load(f)

        # Find and update the user
        user_found = False
        for user in users:
            if user['email'] == email:
                user['Administrator'] = 'Yes'
                user_found = True
                break

        if not user_found:
            return jsonify({'error': 'User not found'}), 404

        # Save changes back to users.json
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=4)
        return jsonify({'message': f'User {email} is now an Administrator'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to accurately update the Subscription shown in User Profile PROFILE HTML

@app.route('/update-subscription', methods=['POST'])
def update_subscription():
    data = request.json
    email = data.get('email')
    new_plan = data.get('subscriptionPlan')

    if not email or not new_plan:
        return jsonify({'error': 'Email and subscription plan are required'}), 400
    users = load_users()
    user_found = False
    for user in users:
        if user['email'] == email:  # Locate the user by email
            user['subscriptionPlan'] = new_plan  # Update the subscriptionPlan
            user_found = True
            break
    if not user_found:
        return jsonify({'error': 'User not found'}), 404
    save_users(users)  # Save updated users list back to JSON
    return jsonify({'message': 'Subscription plan updated successfully'}), 200


# Helper functions
def load_users():
    try:
        with open(USERS_FILE, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.info(f"Error loading users.json: {e}")
        return []

def save_users(users):
    with open(USERS_FILE, 'w') as file:
        json.dump(users, file, indent=4)








# USERS.JSON

def generate_api_key():
    return secrets.token_hex(32)  # Generates a 64-character hex string

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if not data:
        return jsonify({'error': 'No data received'}), 400
    users = load_users()
    if any(user['email'] == data['email'] for user in users):
        return jsonify({'error': 'Email is already registered.'}), 400
    api_key = generate_api_key()
    new_user = {
        "fullName": data['fullName'],
        "email": data['email'],
        "newPassword": data['newPassword'],
        "api_key": api_key,
        "usage": 0.0,
        "usage_denominator": 5, #Example denominator
        "flat_rate_paid": False,
        "Subscribed": "No",  # Default value
        "Unsubscribed": "No",  # Default value
        "Sign Up Status":"Not Yet",  # Default value
        "Administrator": "No",  # Default value
        "AccessEndDate": "2025-01-31",
        "subscriptionPlan": data.get('subscriptionPlan', "")

    }
    users.append(new_user)
    save_users(users)

    # Store email in session
    session['user_email'] = data['email']
    logger.info(f"New user registered and logged in: {data['email']}")  # Debugging
    return jsonify({
        'message': 'Registration successful!',
        'redirect_url': url_for('profile')
    }), 200








# Authenticate API key decorator
def authenticate_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('Authorization')
        if not api_key:
            return jsonify({"error": "API key required"}), 401

        # Find user by API key
        users = load_users()
        user = next((u for u in users if u['api_key'] == api_key), None)
        if not user:
            return jsonify({"error": "Invalid API key"}), 403

        # Pass user data to the endpoint
        return f(user, users, *args, **kwargs)
    return wrapper


# Example API endpoint with usage tracking
@app.route('/api/resource', methods=['GET'])
@authenticate_api_key
def api_resource():
    if 'user_email' not in session:
        return jsonify({"error": "User not logged in"}), 403

    # Load user from session
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)

    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Increment usage
    cost_per_call = 0.10
    user['usage'] += cost_per_call
    save_users(users)
    return jsonify({

        "message": "API request successful.",
        "usage": f"${user['usage']:.2f}",
        "note": "You are charged $0.10 per request."
    }), 200



# LOGOUT FUNCTION

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/login', methods=['GET'])
def serve_login_page():
    reset_success = request.args.get('reset_success')
    if reset_success:
        return render_template('login.html', message="Your password has been reset successfully. Please log in with your new password.")
    return render_template('login.html')

# Route to handle login functionality
@app.route('/login', methods=['POST'])
def handle_login():
    data = request.json
    username = data.get('username')  # Email as username
    password = data.get('password')
    
    # Load users
    users = load_users()
    user = next((u for u in users if u['email'] == username), None)
    if user and user['newPassword'] == password:
        session['user_email'] = user['email']  # Store email in session
        return jsonify({'message': 'Login successful!', 'redirect_url': url_for('handle_upload')}), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401



@app.route('/profile')
def profile():
    return render_template('profile.html')

    
# Counts the Number of Sentences
@app.route('/number_of_sentences')

def index():
    logger.info("Route '/number_of_sentences' accessed.")  # Log that the route was accessed
    results = 2 + 3  # Compute 2 + 3
    logger.info(f"Result of 2 + 2: {results}")  # Log the result of the calculation
    return render_template('upload.html', results=results)  # Render the HTML template and pass `results`






# SHOWS THE LOGO

from flask import send_from_directory
@app.route('/logo.png')

def serve_logo():
    return send_from_directory('.', 'logo.png')  # Serve it from the current directory


# Configure upload folder and file size

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB


# Disclosure words dictionary

DISCLOSURE_WORDS = {}

# Then add this function to load all disclosures from the custom file
def load_all_disclosures():
    """Load all disclosures from the custom disclosures file."""
    global DISCLOSURE_WORDS
    try:
        if os.path.exists(CUSTOM_DISCLOSURES_FILE):
            with open(CUSTOM_DISCLOSURES_FILE, 'r') as f:
                DISCLOSURE_WORDS = json.load(f)
            logger.info(f"Loaded {len(DISCLOSURE_WORDS)} disclosures from {CUSTOM_DISCLOSURES_FILE}")
        else:
            logger.error(f"Custom disclosures file not found: {CUSTOM_DISCLOSURES_FILE}")
            # Initialize with empty dict if file doesn't exist
            DISCLOSURE_WORDS = {}
    except Exception as e:
        logger.error(f"Error loading disclosures: {e}")
        DISCLOSURE_WORDS = {}


# File path for storing custom disclosures persistently
CUSTOM_DISCLOSURES_FILE = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_disclosures.json')

# Ensure the file exists
if not os.path.exists(CUSTOM_DISCLOSURES_FILE):
    with open(CUSTOM_DISCLOSURES_FILE, 'w') as f:
        json.dump({}, f)  # Start with an empty dictionary

@app.route('/add_custom_disclosure', methods=['POST'])
def add_custom_disclosure():
    try:
        data = request.get_json()
        logger.info("Received Data:", data)  # Debugging log
        word = data.get('word', '').strip()
        disclosure_text = data.get('disclosure', '').strip()
        if not word or not disclosure_text:
            return jsonify({"success": False, "error": "Both 'word' and 'disclosure' are required."}), 400
        with open(CUSTOM_DISCLOSURES_FILE, 'r') as f:
            custom_disclosures = json.load(f)
        if word in custom_disclosures:
            return jsonify({"success": False, "error": "Word already exists in custom disclosures."}), 400

        custom_disclosures[word] = disclosure_text
        with open(CUSTOM_DISCLOSURES_FILE, 'w') as f:
            json.dump(custom_disclosures, f, indent=4)

        DISCLOSURE_WORDS[word] = disclosure_text
        return jsonify({"success": True, "message": f"Custom disclosure for '{word}' added successfully."}), 200
    except Exception as e:
        logger.info(f"Error in add_custom_disclosure: {e}")  # Log the error
        return jsonify({"success": False, "error": str(e)}), 500








#Add logging to monitor requests to help debug duplicate requests
    
@app.before_request
def log_request_info():
    logger.info(f"Incoming Request: Path={request.path}, Method={request.method}")
    
#Upload file route
    
@app.route('/', methods=['GET', 'POST'])
def handle_upload():
    # Check if user is logged in
    logged_in_user = session.get('user_email')
    
    # For POST requests, always process the form submission regardless of login status
    if request.method == 'POST':
        # Handle file upload
        distribution_method = request.form.get('distribution_method')
        logger.info(f"Selected distribution method: {distribution_method}")

        uploaded_file = request.files.get('file')
        if not uploaded_file:
            logger.info("No file part in request")  # Debugging message
            return 'No file part', 400
        elif uploaded_file.filename == '':
            logger.info("No selected file")  # Debugging message
            return 'No selected file', 400
        else:
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            logger.info(f"File saved at: {file_path}")  # Debugging message

            # Count sentences in the uploaded file
            if file_path.endswith('.pdf'):
                text_by_page = extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text_by_page = extract_text_from_docx(file_path)
            else:
                return "Unsupported file type", 400

            all_text = " ".join([page["text"] for page in text_by_page])
            sentence_count = len(split_into_sentences(all_text))
            update_user_usage(session.get('user_email'), sentence_count)
            logger.info(f"SENTENCE COUNT #: {sentence_count}")  # Debugging message

            # Redirect to process the file after saving
            return redirect(url_for('process_file', filename=filename))
    
    # For GET requests, check login status
    if not logged_in_user:
        # User is not logged in, show intro page
        logger.info("No user logged in, showing intro page")
        return render_template('intro.html')
    else:
        # User is logged in, show the upload page
        logger.info(f"Logged-in user: {logged_in_user}, showing upload page")
        return render_template('upload.html', user_email=logged_in_user)




# Function to capitalize the first letter of each sentence for display purposes

def capitalize_sentences(text):

    # Capitalizes the first letter of each sentence
    sentence_endings = re.compile(r'([.!?]\s*)')
    sentences = sentence_endings.split(text)
    sentences = [s.capitalize() for s in sentences]
    return ''.join(sentences)


def process_page(page_content):
    page_num = page_content.get("page")
    text = page_content.get("text", "")
    
    # Split text into sentences
    sentences = split_into_sentences(text)
    
    # Log the number of sentences and sample sentences
    logger.info(f"Page {page_num}: Split into {len(sentences)} sentences.")
    for idx, sentence in enumerate(sentences, start=1):
        logger.info(f"Page {page_num} Sentence {idx}: {sentence}")
    
    # Perform compliance check on the entire text
    compliance_data = perform_compliance_check(text, page_num=page_num)
    
    flagged_instances = compliance_data.get("flagged_instances", [])
    
    return flagged_instances





# Process file route
@app.route('/process/<filename>', methods=['GET'])
def process_file(filename):
    global processed_files
    logger.info(f"Processing file: {filename}")
    logger.info(f"File size: {os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename))} bytes")

    logger.info(f"[{datetime.utcnow()}] Incoming request: /process/{filename}")

    disclosures = DISCLOSURE_WORDS

    # Initialize processing status
    final_check_status[filename] = False

    if filename in processed_files:
        logger.info(f"Returning cached results for {filename}.")
        final_check_status[filename] = True
        logger.info("CHECKED!")
        time.sleep(1)
        return render_template(
            'results.html',
            text=Markup(processed_files[filename]['text']),
            revised_text=processed_files[filename].get('revised_text', ''),
            disclosures=processed_files[filename]['disclosures'],
            sliced_disclosures=processed_files[filename].get('sliced_disclosures', []),
            finra_analysis=processed_files[filename]['finra_analysis'],
            results=processed_files[filename].get('results', 0)
        )

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        logger.info(f"File {filename} not found.")
        final_check_status[filename] = False
        return render_template('results.html', text="Error: File not found", revised_text="", disclosures=[], sliced_disclosures=[], finra_analysis=[], results=0)

    try:
        # Extract text by page
        if file_path.endswith('.pdf'):
            text_by_page = extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text_by_page = extract_text_from_docx(file_path)
        else:
            final_check_status[filename] = False
            return render_template('results.html', text="Unsupported file type", revised_text="", disclosures=[], sliced_disclosures=[], finra_analysis=[], results=0)

        all_text = " ".join([page["text"] for page in text_by_page])
        logger.info(f"Total text length: {len(all_text)} characters")
        logger.info(f"First 200 characters: {all_text[:200]}")
        
        # Clean the text - remove bullet points and other special characters
        cleaned_text = re.sub(r'[•\u2022]', '', all_text)  # Remove bullet points
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Clean up whitespace
        
        sentences = split_into_sentences(cleaned_text)
        results = len(sentences)

        finra_analysis = []
        for page in text_by_page:
            page_text = page["text"].replace('•', '').replace('\u2022', '')  # Remove bullet points
            page_num = page.get("page")
            
            # Pass the cleaned text string directly
            compliance_data = perform_compliance_check(page_text, page_num)

            if not compliance_data.get("compliant"):
                instances = compliance_data.get("flagged_instances", [])
                for instance in instances:
                    if isinstance(instance, dict) and instance.get("flagged_instance"):
                        # Only add non-empty, valid instances
                        if instance["flagged_instance"].strip() and \
                           instance["flagged_instance"] != "•" and \
                           len(instance["flagged_instance"]) > 1:
                            finra_analysis.append(instance)

        # Cache results - now includes text_by_page
        processed_files[filename] = {
            'text': cleaned_text,
            'text_by_page': text_by_page,  # Store the original text by page
            'revised_text': "",
            'disclosures': disclosures,
            'sliced_disclosures': list(disclosures.values())[:5],
            'finra_analysis': finra_analysis,
            'results': results
        }

        final_check_status[filename] = True
        logger.info("CHECKED!")
        time.sleep(1)

        # Add filename to localStorage for page-specific disclosures
        return render_template(
            'results.html',
            text=Markup(cleaned_text),
            revised_text="",
            disclosures=disclosures,
            sliced_disclosures=list(disclosures.values())[:5],
            finra_analysis=finra_analysis,
            results=results,
            filename=filename  # Pass filename to template
        )

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        final_check_status[filename] = False
        return render_template('results.html', text="Error processing file", revised_text="", disclosures=[], sliced_disclosures=[], finra_analysis=[], results=0)

# Function to remove repeated phrases more effectively.   
def remove_repeated_phrases_v2(text):
    words = text.split()
    cleaned_text = []
    buffer = []
    prev_chunk = []

    for word in words:
        buffer.append(word)
        if len(buffer) >= 5:  # Check for repeated sequences every 5 words
            current_chunk = ' '.join(buffer)
            if prev_chunk and current_chunk == prev_chunk:

                # If repeated, skip appending the duplicate chunk
                buffer = []
            else:

                # Append the chunk and reset the buffer
                cleaned_text.extend(buffer)
                prev_chunk = current_chunk
                buffer = []

    # Append any remaining words in the buffer
    if buffer:
        cleaned_text.extend(buffer)
    return ' '.join(cleaned_text)



# Load the spaCy model for English (you need to run 'python -m spacy download en_core_web_sm' once)
nlp = spacy.load("en_core_web_sm")

# Add the sentencizer to improve sentence splitting
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
    

# Function to check contextual similarity using NLP
USE_SIMILARITY_CHECK = False  # Set to True to enable the similarity check

def is_contextually_similar(text, keyword):
    if not USE_SIMILARITY_CHECK:

        # Placeholder logic or always return False if similarity check is disabled
        return False

    # Use spacy to compare similarity between the document and the keyword
    doc = nlp(text)
    keyword_doc = nlp(keyword)
    similarity = doc.similarity(keyword_doc)

    # Define a similarity threshold; adjust as needed for sensitivity
    threshold = 0.7  # 70% similar
    return similarity >= threshold

# Function to check similarity ratio using SequenceMatcher
def is_similar_ratio(text, keyword):
    return SequenceMatcher(None, text, keyword).ratio() >= 0.7  # Adjust threshold if needed

def normalize_disclosure(text):
    return ' '.join(text.split()).lower()  # Normalize spacing and case




# Function to analyze text for trigger and disclosure words with contextual recognition
def analyze_text(text):
    """
    Analyze text for trigger and disclosure words with contextual recognition.
    Returns only disclosures that match keywords in the text.
    """
    logger.info(f"analyze_text called with: '{text[:50]}...'")
    
    # Load standard disclosures from disclosures.json
    standard_disclosures = {}
    DISCLOSURES_FILE = os.path.join(os.path.dirname(__file__), 'disclosures.json')
    
    try:
        if os.path.exists(DISCLOSURES_FILE):
            with open(DISCLOSURES_FILE, 'r') as f:
                disclosures_data = json.load(f)
                
                # Log the structure of the loaded data
                logger.info(f"Loaded disclosures data type: {type(disclosures_data)}")
                if isinstance(disclosures_data, list):
                    logger.info(f"List length: {len(disclosures_data)}")
                    if len(disclosures_data) > 0:
                        logger.info(f"First item type: {type(disclosures_data[0])}")
                
                # Handle different formats of disclosures.json
                if isinstance(disclosures_data, list) and len(disclosures_data) > 0:
                    standard_disclosures = disclosures_data[0] if isinstance(disclosures_data[0], dict) else {}
                else:
                    standard_disclosures = disclosures_data
                
                logger.info(f"Loaded {len(standard_disclosures)} standard disclosures")
    except Exception as e:
        logger.error(f"Error loading standard disclosures: {e}")
    
    # Load custom disclosures
    custom_disclosures = {}
    CUSTOM_DISCLOSURES_FILE = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_disclosures.json')
    
    try:
        if os.path.exists(CUSTOM_DISCLOSURES_FILE):
            with open(CUSTOM_DISCLOSURES_FILE, 'r') as f:
                custom_disclosures = json.load(f)
                logger.info(f"Loaded {len(custom_disclosures)} custom disclosures")
    except Exception as e:
        logger.error(f"Error loading custom disclosures: {e}")
    
    # Merge disclosures, with custom taking precedence
    all_disclosures = {**standard_disclosures, **custom_disclosures}
    logger.info(f"Combined disclosure dictionary has {len(all_disclosures)} items")
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Track matched disclosures and keywords
    matched_disclosures = []
    matched_keywords = []
    
    # Start with general disclosure
    general_disclosure = "Content in this material is for general information only and not intended to provide specific advice or recommendations for any individual."
    matched_disclosures.append(general_disclosure)
    logger.info(f"Added general disclosure by default")
    
    # Check each keyword for a match in the text
    for keyword, disclosure in all_disclosures.items():
        keyword_lower = keyword.lower()
        
        # If the keyword is found in the text
        if keyword_lower in text_lower:
            matched_keywords.append(keyword)
            
            # Add the disclosure if not already present
            if disclosure not in matched_disclosures:
                matched_disclosures.append(disclosure)
                logger.info(f"Matched keyword '{keyword}' to disclosure: {disclosure[:30]}...")
    
    # Handle special case for 'investing'
    if "invest" in text_lower or "investing" in text_lower:
        investing_disclosure = "Investing involves risk including loss of principal. No strategy assures success or protects against loss."
        if investing_disclosure not in matched_disclosures:
            matched_disclosures.append(investing_disclosure)
            logger.info(f"Added investing disclosure based on 'invest/investing' keyword")
    
    # Handle special case for 'mutual funds'
    if "mutual fund" in text_lower or "funds" in text_lower:
        mutual_fund_disclosure = all_disclosures.get("Mutual Fund", "")
        if mutual_fund_disclosure and mutual_fund_disclosure not in matched_disclosures:
            matched_disclosures.append(mutual_fund_disclosure)
            logger.info(f"Added mutual fund disclosure based on 'funds' keyword")
    
    logger.info(f"Found {len(matched_keywords)} matching keywords: {matched_keywords}")
    logger.info(f"Returning {len(matched_disclosures)} matched disclosures")
    
    return text, matched_disclosures




# Route for turning text into pdf on upload page
from fpdf import FPDF  # Add this import at the top of app.py if not already present
from flask import jsonify, url_for

@app.route("/convert_text_to_pdf", methods=["POST"])
def convert_text_to_pdf():
    data = request.get_json()
    text_content = data.get("text_content", "")
    if text_content:

        # Save PDF in the uploads folder in the Project directory
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "converted_text.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        # Add the text content to the PDF file
        pdf.multi_cell(0, 10, text_content)
        pdf.output(pdf_path)

        # Return URL pointing to the newly created PDF
        pdf_url = url_for('serve_file', filename="converted_text.pdf", _external=True)
        return jsonify({"pdf_url": pdf_url})
    return "Error converting text to PDF", 400


@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/analyze_file_content", methods=["POST"])
def analyze_file_content():

    # Check if the request has a file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save file temporarily for processing, or use in-memory processing
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_analysis.pdf")
    file.save(temp_path)
    
    # Example: perform your analysis function on the file (process text, run compliance check, etc.)

    # This is where you call your actual file processing logic
    results = perform_file_analysis(temp_path)  # Replace with your actual analysis function

    # Clean up by removing the temp file, if saved
    os.remove(temp_path)

    # Return the analysis results as JSON
    return jsonify(results)









# Progress Loading Bar
@app.route('/progress')
def progress():

    def generate():
        for i in range(0, 101, 10):  # Simulating progress
            yield f"data: {i}\n\n"
            time.sleep(0.5)  # Simulated delay
    return Response(stream_with_context(generate()), mimetype='text/event-stream')








# Recycle Button Function

# Load the compliance examples from your JSON file
def load_compliance_examples():
    try:
        with open('fcd.json', 'r') as f:
            data = json.load(f)
            
            # If data is already a list, use it directly
            if isinstance(data, list):
                examples = data
            # If data is a dictionary with numbered keys (like {"0": {...}, "1": {...}, ...})
            elif isinstance(data, dict):
                examples = list(data.values())
            else:
                logger.error(f"Unexpected JSON format in fcd.json")
                return []
                
            # Filter to ensure we only use entries with both non_compliant and compliant text
            valid_examples = [
                example for example in examples 
                if "non_compliant" in example and "compliant" in example
            ]
            
            logger.info(f"Loaded {len(valid_examples)} valid compliance examples from fcd.json")
            return valid_examples
            
    except Exception as e:
        logger.error(f"Error loading compliance examples: {e}")
        return []

    
# Function to calculate text similarity
def similarity_score(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Add this at the top of your file with other imports/globals
last_alternatives = {}  # Simple dict to track the most recent alternative for each flagged text

@app.route('/generate_new_alternative', methods=['POST'])
def generate_new_alternative():
    data = request.get_json()
    flagged_text = data.get("flagged_instance", "").strip()
    if not flagged_text:
        return jsonify({"new_alternative": None, "feedback": "No flagged instance provided."}), 400

    # Get the last alternative shown for this flagged text, if any
    last_alternative = last_alternatives.get(flagged_text)
    
    try:
        # Load compliance examples from JSON
        examples = load_compliance_examples()
        if not examples:
            # Fallback if JSON loading fails
            return jsonify({
                "new_alternative": "We strive to help clients pursue their investment objectives while understanding that all investments involve risk.",
                "rationale": "Modified to use more balanced language and avoid promissory statements."
            })
        
        # 1. First approach: Find similar non-compliant examples
        similarity_threshold = 0.6
        similar_examples = []
        
        for example in examples:
            non_compliant = example.get("non_compliant", "")
            sim_score = similarity_score(flagged_text, non_compliant)
            if sim_score > similarity_threshold:
                similar_examples.append((example, sim_score))
        
        # Sort by similarity score (descending)
        similar_examples.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 matches if available
        top_matches = similar_examples[:3] if len(similar_examples) >= 3 else similar_examples
        
        if top_matches:
            # Filter out the last alternative if it's in the top matches
            filtered_matches = [(ex, score) for ex, score in top_matches 
                              if ex.get("compliant") != last_alternative]
            
            # Use filtered matches if available, otherwise use original top matches
            if filtered_matches:
                selected = random.choice(filtered_matches)
            else:
                # If all matches were filtered out, just choose one different from the last
                if len(top_matches) > 1:
                    # Try to pick a different one if there's more than one match
                    for _ in range(3):  # Try a few times to get a different one
                        selected = random.choice(top_matches)
                        if selected[0].get("compliant") != last_alternative:
                            break
                else:
                    # Just use what we have if only one match
                    selected = top_matches[0]
                    
            new_alternative = selected[0].get("compliant")
            logger.info(f"Found similar example with score {selected[1]:.2f}")
            
        else:
            # 2. Second approach: Keyword matching
            # Extract key words from flagged text (removing common words)
            words = [w.lower() for w in re.findall(r'\b\w+\b', flagged_text) 
                    if len(w) > 3 and w.lower() not in ('with', 'that', 'this', 'from', 'they', 'will', 
                                                        'have', 'been', 'were', 'being', 'does', 'where',
                                                        'which', 'their', 'there', 'about', 'would')]
            
            keyword_matches = []
            for example in examples:
                non_compliant = example.get("non_compliant", "").lower()
                # Count how many keywords match
                matches = sum(1 for word in words if word in non_compliant)
                if matches > 0:
                    keyword_matches.append((example, matches))
            
            # Sort by number of matching keywords (descending)
            keyword_matches.sort(key=lambda x: x[1], reverse=True)
            
            if keyword_matches:
                # Get up to 3 best keyword matches
                best_keyword_matches = keyword_matches[:3]
                
                # Filter out the last alternative
                filtered_matches = [(ex, matches) for ex, matches in best_keyword_matches 
                                  if ex.get("compliant") != last_alternative]
                
                # Use filtered matches if available, otherwise use original matches
                if filtered_matches:
                    selected = random.choice(filtered_matches)
                else:
                    # Try to pick a different one if possible
                    if len(best_keyword_matches) > 1:
                        for _ in range(3):  # Try a few times
                            selected = random.choice(best_keyword_matches)
                            if selected[0].get("compliant") != last_alternative:
                                break
                    else:
                        selected = best_keyword_matches[0]
                        
                new_alternative = selected[0].get("compliant")
                logger.info(f"Found keyword match with {selected[1]} matching keywords")
            else:
                # 3. Final fallback: Select a random compliant example
                # Try to pick one different from last time
                for _ in range(5):  # Try a few times
                    random_example = random.choice(examples)
                    new_alternative = random_example.get("compliant")
                    if new_alternative != last_alternative:
                        break
                logger.info("No good matches found, using random example")
        
        # Store this as the last alternative for next time
        last_alternatives[flagged_text] = new_alternative
        
        # Generate a rationale
        rationale = "Modified to use more balanced language and avoid promissory statements."
        
        # Update user usage if logged in
        if 'user_email' in session:
            user_email = session['user_email']
            users = load_users()
            user = next((u for u in users if u['email'] == user_email), None)
            if user:
                user['usage'] += 0.05  # Add $0.05 for this recycle task
                save_users(users)
                logger.info(f"Updated usage for {user_email}: ${user['usage']:.2f}")

        return jsonify({"new_alternative": new_alternative, "rationale": rationale})
        
    except Exception as e:
        logger.error(f"Error generating alternative: {e}")
        return jsonify({
            "new_alternative": "We aim to help clients create a plan that aligns with their investment objectives.",
            "rationale": "Modified to use more balanced language and avoid promissory statements."
        })


    
#??? 
def fix_compliance_issues(text):
    """Create a clean, coherent compliant alternative."""
    
    # First, analyze what type of statement this is
    text = text.lower().strip()
    logger.info(f"Original text: {text}")
    
    # Check if this is about investing/making money
    if any(term in text for term in ['invest', 'money', 'return', 'profit', 'earn']):
        # This is likely about investment returns
        if "provide to" in text or "aim to provide" in text or "seek to provide" in text:
            # Fix awkward phrasing from previous replacements
            alternative = "I aim to help you pursue your financial goals through investing, but this involves risk."
        elif "will" in text:
            alternative = "I may help you pursue your financial goals through investing, but this involves risk."
        elif "guarantee" in text or "guaranteed" in text:
            alternative = "I aim to help you pursue your financial goals through investing, but results cannot be guaranteed."
        elif "make you money" in text:
            alternative = "I strive to help you pursue your financial objectives through investing, though all investments involve risk."
        else:
            # General investment statement
            alternative = "I aim to help you pursue your financial goals through investing, while understanding that all investments involve risk."
    
    # Check if this is about market prediction
    elif any(term in text for term in ['market', 'stock', 'bond', 'price', 'rise', 'fall', 'increase', 'decrease']):
        if "will" in text:
            alternative = "The market may experience changes based on various factors, but future performance cannot be predicted."
        else:
            alternative = "Markets may fluctuate based on various factors, and past performance does not guarantee future results."
    
    # General case for other types of statements
    else:
        # Replace key problematic terms
        text = text.replace("will", "may")
        text = text.replace("always", "often")
        text = text.replace("never", "rarely")
        text = text.replace("guarantee", "aim for")
        text = text.replace("best", "strong")
        
        # Fix any awkward phrasing from previous replacements
        text = text.replace("aim for to", "aim to")
        text = text.replace("provide to", "help")
        text = text.replace("seek to provide to", "strive to")
        
        # Ensure proper sentence structure
        alternative = text.capitalize()
        if not alternative.endswith(('.', '!', '?')):
            alternative += '.'
    
    logger.info(f"Generated alternative: {alternative}")
    return alternative




#???
def parse_compliance_response(response_text):
    # This variable tracks overall compliance status (top-level),
    # which you may or may not need depending on your use case.
    top_level_compliance_status = None
    
    message = ""
    flagged_instances = []
    current_instance = {}

    # Split the response into lines for processing
    lines = response_text.split('\n')
    
    for line in lines:
        line = line.strip()

        # Check if this is the top-level "Compliance Status: ..." line
        # (In some prompt formats, you might have a line like "Compliance Status: Compliant"
        #  at the very top. If you do not have a top-level compliance status, you can remove this.)
        if line.startswith("Compliance Status:"):
            top_level_compliance_status = line.split(":", 1)[1].strip().lower()

        elif line.startswith("Message:"):
            message = line.split(":", 1)[1].strip()

        elif line.startswith("Flagged Instances:"):
            # "Flagged Instances:" is just a header line, skip it
            continue

        elif re.match(r'^\d+\.\s+"(.+)"$', line):
            # We've reached a new flagged instance.
            # 1) If we have a previously built instance, append it (if it meets our filter).
            if current_instance:
                # Only append if "Mostly Compliant" or "Non-Compliant"
                status_val = current_instance.get("compliance_status", "").lower()
                if status_val in ["non-compliant"]:
                    flagged_instances.append(current_instance)
            
            # 2) Start building the new instance
            current_instance = {}
            match = re.match(r'^\d+\.\s+"(.+)"$', line)
            if match:
                current_instance["flagged_instance"] = match.group(1)

        elif line.startswith("- Compliance Status:"):
            # e.g. "- Compliance Status: Mostly Compliant."
            # remove trailing period if present and make it lowercase for consistency
            status_text = line.split(":", 1)[1].strip().rstrip(".").lower()
            current_instance["compliance_status"] = status_text

        elif line.startswith("- Specific Compliant Alternative:"):
            # e.g. "- Specific Compliant Alternative: "Invest in an IRA...""
            alt_text = line.split(":", 1)[1].strip().strip('"')
            current_instance["specific_compliant_alternative"] = alt_text

        elif line.startswith("- Rationale:"):
            # e.g. "- Rationale: "This is too promissory..."
            rationale_text = line.split(":", 1)[1].strip().strip('"')
            current_instance["rationale"] = rationale_text

        else:
            # Capture any unexpected lines for debugging
            if line and not line.startswith("---"):
                logger.warning(f"Unrecognized line format: {line}")

    # After the loop, we may have one last instance to append
    if current_instance:
        status_val = current_instance.get("compliance_status", "").lower()
        if status_val in ["non-compliant"]:
            flagged_instances.append(current_instance)

    # Decide how you want to handle overall compliance:
    # We can define top-level "compliant" as exactly "compliant",
    # or you might want to derive it from flagged instances instead.
    # Here we directly follow the old approach: 
    return {
        "compliant": (top_level_compliance_status == "compliant"),
        "message": message,
        "flagged_instances": flagged_instances
    }




@app.route('/check_custom_alternative', methods=['POST'])
def check_custom_alternative():
    data = request.get_json()
    custom_text = data.get("custom_text", "").strip()
    
    if not custom_text:
        logger.error("No text provided for compliance check.")
        return jsonify({"error": "No text provided for compliance check."}), 400
    
    try:
        global BERT_MODEL, BERT_TOKENIZER
        if BERT_MODEL is None or BERT_TOKENIZER is None:
            if not initialize_bert():
                raise Exception("Failed to initialize BERT model")
        
        # Load compliance examples for better alternatives and rationales
        compliance_examples = load_compliance_examples()
        
        # Split text into sentences for BERT analysis
        sentences = split_into_sentences(custom_text)
        flagged_instances = []
        
        for sentence in sentences:
            # Skip very short or empty sentences
            if not sentence.strip() or len(sentence.split()) < 3:
                continue
            
            # Prepare input for BERT
            inputs = BERT_TOKENIZER(sentence, 
                                  return_tensors="pt",
                                  truncation=True,
                                  max_length=512,
                                  padding=True)
        
            # Make prediction using BERT
            with torch.no_grad():
                outputs = BERT_MODEL(**inputs)
                probabilities = softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            logger.info(f"BERT prediction for sentence: {prediction}, Confidence: {confidence:.2f}")
            
            # If BERT predicts non-compliant (class 1) with sufficient confidence
            if prediction == 1 and confidence > 0.7:
                # Generate a compliant alternative with enhanced rationale
                compliant_text, rule, rationale = generate_compliant_alternative(
                    sentence, confidence, compliance_examples)
                
                instance = {
                    "flagged_instance": sentence,
                    "compliance_status": "non-compliant",
                    "specific_compliant_alternative": compliant_text,
                    "rationale": rationale,  # Use the enhanced rationale
                    "confidence": f"{confidence:.1%}"
                }
                flagged_instances.append(instance)
        
        # Update user usage if needed
        if 'user_email' in session:
            update_user_usage(session['user_email'], usage_increment=0.05)
        
        # Determine overall compliance based on flagged instances
        is_compliant = len(flagged_instances) == 0
        
        return jsonify({
            "compliant": is_compliant,
            "message": "No issues identified." if is_compliant else "This text may not comply with FINRA regulations.",
            "flagged_instances": flagged_instances
        }), 200
        
    except Exception as e:
        logger.error(f"Error during custom text compliance check: {e}")
        return jsonify({
            "compliant": False,
            "error": "An error occurred during compliance checking.",
            "message": str(e)
        }), 500


# "/" Route

@app.route("/", methods=["POST"])

def process_form():
    
    # Simulate response for testing
    data = "Processing complete."
    response = Response(data, mimetype="text/plain")
    response.headers["Content-Length"] = str(len(data))
    return response




# GENERATING DISCLOSURES FUNCTION
@app.route("/generate_disclosures", methods=["POST"])
def generate_disclosures():
    data = request.get_json()
    revised_text = data.get("text", "")
    
    # Add detailed logging
    logger.info(f"generate_disclosures called with text: '{revised_text[:50]}...'")

    # Call your existing disclosure logic for the revised text
    analyzed_text, disclosures = analyze_text(revised_text)
    
    # Log what analyze_text returned
    logger.info(f"analyze_text returned {len(disclosures)} disclosures")
    for i, disc in enumerate(disclosures):
        logger.info(f"Disclosure {i+1}: {disc[:50]}...")

    # Load scenarios from scenarios.json
    with open("scenarios.json") as f:
        scenarios = json.load(f)

    # Determine the logged-in user
    logged_in_user = request.headers.get("Logged-In-User", "Unknown")
    logger.info(f"Logged-in user: {logged_in_user}")

    # Find the corresponding scenario for the user
    assigned_scenario = scenarios.get(logged_in_user, "No scenario assigned.")
    logger.info(f"Assigned scenario: {assigned_scenario[:50]}...")

    # Append specific key disclosures in a defined order
    investing_disclosure = "Investing involves risk including loss of principal. No strategy assures success or protects against loss."
    general_disclosure = "Content in this material is for general information only and not intended to provide specific advice or recommendations for any individual."
    
    # IMPORTANT: Only use the disclosures returned by analyze_text, don't add more
    # final_disclosures = [investing_disclosure, general_disclosure] + disclosures
    final_disclosures = disclosures  # Use ONLY what analyze_text returns
    final_disclosures = list(dict.fromkeys(final_disclosures))  # Remove duplicates if any
    
    logger.info(f"Returning {len(final_disclosures)} final disclosures")

    return jsonify({
        "investing_disclosure": investing_disclosure,
        "disclosures": final_disclosures,
        "assigned_scenario": assigned_scenario,
    })


# Function to upload a file (to be archived or for processing... not sure?)

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logger.info(f"File uploaded and saved to: {file_path}")

    # Return the filename for progress tracking
    return jsonify({'filename': filename}), 200



# CHECKING QUICK TEXT

@app.route('/check_quick_text', methods=['POST'])
def check_quick_text():
    data = request.get_json()
    custom_text = data.get("text", "").strip()
    
    if not custom_text:
        logger.error("No text provided for quick compliance check.")
        return jsonify({"error": "No text provided."}), 400
    
    try:
        global BERT_MODEL, BERT_TOKENIZER
        if BERT_MODEL is None or BERT_TOKENIZER is None:
            if not initialize_bert():
                raise Exception("Failed to initialize BERT model")
        
        # Split text into sentences for BERT analysis
        sentences = split_into_sentences(custom_text)
        flagged_instances = []
        
        for sentence in sentences:
            # Skip very short or empty sentences
            if not sentence.strip() or len(sentence.split()) < 3:
                continue
            
            # Prepare input for BERT
            inputs = BERT_TOKENIZER(sentence, 
                                  return_tensors="pt",
                                  truncation=True,
                                  max_length=512,
                                  padding=True)
        
            # Make prediction using BERT
            with torch.no_grad():
                outputs = BERT_MODEL(**inputs)
                probabilities = softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            logger.info(f"BERT prediction for sentence: {prediction}, Confidence: {confidence:.2f}")
            
            # If BERT predicts non-compliant (class 1) with sufficient confidence
            if prediction == 1 and confidence > 0.7:
                # Create a simple alternative by making basic modifications
                modified_text = sentence.replace("will", "may").replace("guarantee", "aim to provide")
                
                instance = {
                    "flagged_instance": sentence,
                    "compliance_status": "non-compliant",
                    "specific_compliant_alternative": modified_text,
                    "rationale": "This text may not comply with FINRA regulations. Try using more balanced language and avoid promissory statements.",
                    "confidence": f"{confidence:.1%}"
                }
                flagged_instances.append(instance)
        
        # Update user usage if needed
        if 'user_email' in session:
            update_user_usage(session['user_email'], usage_increment=0.05)
        
        # Determine overall compliance based on flagged instances
        is_compliant = len(flagged_instances) == 0
        
        return jsonify({
            "compliant": is_compliant,
            "message": "No issues identified." if is_compliant else "Compliance issues found.",
            "flagged_instances": flagged_instances
        }), 200
        
    except Exception as e:
        logger.error(f"Error during quick text compliance check: {e}")
        return jsonify({
            "compliant": False,
            "error": "An error occurred during compliance checking.",
            "message": str(e)
        }), 500


# LOADING BAR FUNCTIONALITY AND RULES

# Dictionary to track final check status per file
final_check_status = {}

@app.route('/progress/<filename>', methods=['GET'])
def progress_stream(filename):
    def generate_progress():
        try:
            logger.info(f"Progress endpoint called with filename: {filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.isfile(file_path):
                yield f"data: Error: File not found\n\n"
                return

            # Dynamically determine the number of sentences
            text_by_page = extract_text_from_pdf(file_path) if filename.endswith('.pdf') else extract_text_from_docx(file_path)
            all_text = " ".join([page["text"] for page in text_by_page])
            sentences = split_into_sentences(all_text)
            flagged_instances = []  # Define flagged instances if necessary
            total_sentences = len(sentences)

            if total_sentences == 0:
                yield f"data: Error: No sentences to process\n\n"
                return

            # Simulate progress to 60%
            increment = 60 / total_sentences
            for i, sentence in enumerate(sentences, start=1):
                time.sleep(0.9)  # Simulate processing time
                progress = i * increment
                yield f"data: {progress:.2f}\n\n"

                # Simulate flagged instance detection
                if i <= len(flagged_instances):
                    yield f"data: Flagged Instance: {flagged_instances[i - 1]}\n\n"

                # Check if the final check is completed during sentence processing
                if final_check_status.get(filename, False):
                    yield "data: 100.00\n\n"
                    logger.info(f"Final check completed for {filename}. Progress set to 100% during sentence processing.")
                    break  # Stop further sentence processing as the final check is done

            # Wait for the processing to complete in /process/<filename>
            progress = 60  # Start at 60%
            while not final_check_status.get(filename, False):
                progress += 2  # Increment progress by 2% each iteration
                if progress > 90:  # Cap progress at 90%
                    progress = 90
                logger.info(f"Waiting for final check completion for {filename}... Progress: {progress}%")
                yield f"data: {progress}\n\n"  # Send progress to the frontend
                time.sleep(2.5)  # Match frontend timing

            # Check if the final check is completed during sentence processing
                if final_check_status.get(filename, False):
                    yield "data: 100.00\n\n"
                    logger.info(f"Final check completed for {filename}. Progress set to 100% during sentence processing.")
                    break  # Stop further sentence processing as the final check is done

            # Emit "Processing Complete!" once CHECKED! is logged
            logger.info(f"Final check completed for {filename}. Sending completion signal.")
            yield "data: 100.00\n\n"
            yield "data: Processing Complete!\n\n"

        except Exception as e:
            logger.error(f"Error in progress_stream: {e}")
            yield f"data: Error: {str(e)}\n\n"

    return Response(generate_progress(), content_type='text/event-stream')

# Terminal Logs
@app.route('/terminal_logs', methods=['GET'])
def terminal_logs():

    def generate_logs():
        try:
            with open("uploads/app.log", "r") as log_file:
                log_file.seek(0, os.SEEK_END)  # Start at the end of the file
                while True:
                    line = log_file.readline()
                    if line:
                        yield f"data: {line.strip()}\n\n"
                    time.sleep(0.1)
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    return Response(stream_with_context(generate_logs()), mimetype='text/event-stream')

if __name__ == "__main__":
    # Load all disclosures from the custom file
    load_all_disclosures()
    
    # Initialize BERT model
    if not initialize_bert():
        logger.error("Failed to initialize BERT model")
    else:
        logger.info("BERT model initialized successfully")
    
    # Get the port from the environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
