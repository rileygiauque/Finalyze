# APP.PY

# FROM AND IMPORT CODES

import math
import time
import traceback
from pydub import AudioSegment

import speech_recognition as sr
import moviepy.editor as mp

import json
from flask import jsonify, request
import json
from flask import Flask, render_template, redirect, session, request, jsonify
import os
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
import traceback
import requests

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
import stripe
import openai

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax


flask_logger = logging.getLogger('werkzeug')
flask_logger.setLevel(logging.WARNING)  # Change to INFO if you want to capture more details

# At the global level of your app.py
BERT_MODEL = None
BERT_TOKENIZER = None
MODEL_LOADED = False
MODEL_PATH = 'finra_compliance_model.pth'  # Update this to your actual model path

# Add this after your global variable declarations
def get_bert_model():
    global BERT_MODEL, BERT_TOKENIZER, MODEL_LOADED
    if not MODEL_LOADED:
        # Log start of model loading
        app.logger.info("Initializing BERT model for compliance check...")
        app.logger.info(f"Loading model file: {MODEL_PATH}")
        
        # Get file info for logging
        if os.path.exists(MODEL_PATH):
            modified_time = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
            app.logger.info(f"Last modified: {modified_time}")
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            app.logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Load model and tokenizer
        BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        BERT_MODEL = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        BERT_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        BERT_MODEL.eval()
        MODEL_LOADED = True
    
    return BERT_MODEL, BERT_TOKENIZER


# Add this near the top of your file with other global variables
session_token_usage = {
    'prompt_tokens': 0, 
    'completion_tokens': 0,
    'total_tokens': 0,
    'total_cost': 0.0,
    'api_calls': 0,
    'start_time': time.time()  # Track when the session started
}


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

#Sendgrid Key
os.environ['SENDGRID_API_KEY'] = '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'

# MY CHATGPT API KEY
#openai.api_key = "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"

# My Deepseek API Key
DEEPSEEK_API_KEY = "sk-236ec31139a5435fa2b4720c53601c09"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # adjust based on actual Deepseek API endpoint


# MY ANTHROPIC KEY

SYSTEM_MESSAGE = """You are a compliance assistant specializing in FINRA's social media communication rules and regulations. Analyze the following text for compliance with these EXACT rules:

RULE 1: All communications must be fair, balanced, and complete and not omit material information.
Violations include:
- Presenting benefits without equal emphasis on risks
- Omitting key product/strategy limitations
- Discussing returns without mentioning potential losses
- Highlighting advantages without corresponding disadvantages
- Selective disclosure of performance periods

RULE 2: False, misleading, promissory, exaggerated, or unwarranted statements/claims are prohibited.
Violations include:
- "Will" statements about performance or outcomes
- Guarantees of any kind
- Absolute statements ("always," "never," "best")
- Unsubstantiated claims about performance
- Promises of specific results

RULE 3: Material information may not be buried in footnotes.
Violations include:
- Important disclosures in smaller text
- Key risks relegated to footnotes
- Critical information placed inconspicuously
- Material facts separated from main claims

RULE 4: Statements must be clear and provide balanced treatment of risks/benefits.
Violations include:
- Complex terminology without explanation
- Understated risks
- Overstated benefits
- Imbalanced risk-reward presentations
- Unclear or ambiguous statements

RULE 5: Communications must be appropriate for the audience.
Violations include:
- Technical jargon without explanation
- Complex strategies without context
- Inappropriate risk levels
- Unsuitable recommendations
- Mismatched complexity

ANALYSIS INSTRUCTIONS:
1. Compare text EXACTLY against these rules
2. Flag ANY instance that matches the violation examples
3. For each flagged instance, provide:
   - The exact non-compliant text
   - Which specific rule it violates
   - A clear rationale
   - A compliant alternative

RESPONSE FORMAT:
For each violation, respond in this exact format:
---
Flagged Instance: "[exact text]"
Rule Violated: [specific rule number and description]
Compliance Status: [Non-Compliant/Partially Compliant]
Specific Compliant Alternative: "[alternative text]"
Rationale: "[explanation of violation and why alternative complies]"
---

ALTERNATIVE TEXT REQUIREMENTS:
- Replace "will" with "may"
- Remove absolute statements
- Add appropriate qualifiers
- Balance risk and reward
- Maintain original meaning
- Keep similar length
- Use clear language

If no violations are found, respond ONLY with:
"Compliance Check Completed. No issues found."

IMPORTANT: Be consistent in flagging similar violations across different texts. When in doubt, flag the instance."""

# Initialize client
#openai.api_key = "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"  # Replace with your actual OpenAI API key




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




# PROHIBITED WORDS (in control panel)
def load_prohibited_words_db():
    """Load prohibited words from PostgreSQL only (no JSON). Returns a list of words."""

    try:
        conn = psycopg2.connect(
            dbname="postgresql_instance_free",
            user="postgresql_instance_free_user",
            password="bz3SdnKi6g6TRdM4j1AtE2Ash8VNgiQO",
            host="dpg-cts4psa3esus73dn1cn0-a.oregon-postgres.render.com",
            port="5432"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT word FROM prohibited_words;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Convert rows → list of words
        return [row[0] for row in rows]
    except Exception as e:
        logger.info(f"Error loading prohibited words from DB: {e}")
        return []



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



def segment_transcription_into_sentences(transcribed_text):
    """
    Split transcribed text into properly formatted sentences with focus on natural flow.
    """
    # Clean the text
    text = transcribed_text.strip()
    
    if not text:
        return []
    
    # First, handle existing punctuation properly
    # This makes sure we preserve any actual sentence breaks in the transcription
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after periods
    
    # Fix common speech recognition issues:
    # 1. Remove unnecessary periods that often appear in speech recognition
    text = re.sub(r'\b(mr|mrs|ms|dr)\.\s', r'\1 ', text, flags=re.IGNORECASE)  # Remove title periods
    text = re.sub(r'\s+right\.\s+now', r' right now', text, flags=re.IGNORECASE)  # Fix "right. now"
    
    # 2. Add proper periods at clear sentence boundaries
    text = re.sub(r'([.!?])\s+([A-Z])', r'\1 \2', text)  # Ensure sentence boundaries are preserved
    
    # 3. Add periods at clear transition points
    transitions = [
        r'(\w+)\s+(So today|Anyway|Alright|Now let\'s)\b',
        r'(\w+)\s+(Let me|I want to|We need to)\b',
        r'(\w+[\.\!\?]*)\s+(Meanwhile|Furthermore|However|Therefore|Thus|Hence)\b'
    ]
    for pattern in transitions:
        text = re.sub(pattern, r'\1. \2', text, flags=re.IGNORECASE)
    
    # 4. Mark clear standalone interjections
    interjections = [r'\b(No|Yes|Yeah|Hey|Hi|Hello|Thanks|Thank you)\b\s+([A-Z])']
    for pattern in interjections:
        text = re.sub(pattern, r'\1. \2', text, flags=re.IGNORECASE)
    
    # Use a simplified spaCy approach if available
    sentences = []
    if 'nlp' in globals():
        try:
            # Process with spaCy and apply minimal post-processing
            doc = nlp(text)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                
                # Apply basic sentence formatting
                if not sent_text[-1] in ['.', '!', '?']:
                    sent_text += '.'
                sent_text = sent_text[0].upper() + sent_text[1:] if sent_text and sent_text[0].islower() else sent_text
                
                # Only break very long sentences (>30 words)
                if len(sent_text.split()) > 30:
                    parts = simple_break_long_sentence(sent_text)
                    sentences.extend(parts)
                else:
                    sentences.append(sent_text)
            
            if len(sentences) >= 2:
                logger.info(f"Used spaCy to segment text into {len(sentences)} sentences")
                return sentences
        except Exception as e:
            logger.error(f"Error using spaCy for sentence segmentation: {e}")
    
    # Fallback approach if spaCy isn't available or didn't work as expected
    
    # Manual sentence detection with careful boundary handling
    # This simpler approach looks for sentence boundaries without over-processing
    manual_sentences = []
    
    # Split by sentence endings
    raw_segments = re.split(r'(?<=[.!?])\s+', text)
    segments = []
    
    # Clean up segments and handle potential false breaks
    current_segment = ""
    for segment in raw_segments:
        segment = segment.strip()
        if not segment:
            continue
            
        # Check if this looks like a false break (e.g., "Mr. Smith")
        if current_segment and re.search(r'\b(mr|mrs|ms|dr|inc|ltd|co|etc|vs|fig|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\.$', 
                                          current_segment, re.IGNORECASE):
            current_segment += " " + segment
        else:
            if current_segment:
                segments.append(current_segment)
            current_segment = segment
    
    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)
    
    # Process each cleaned segment
    for segment in segments:
        if len(segment.split()) <= 25:
            # For normal length segments, just format and add
            if not segment[-1] in ['.', '!', '?']:
                segment += '.'
            segment = segment[0].upper() + segment[1:] if segment and segment[0].islower() else segment
            manual_sentences.append(segment)
        else:
            # For longer segments, try to break at natural points
            parts = simple_break_long_sentence(segment)
            manual_sentences.extend(parts)
    
    # Final safety check
    if not manual_sentences and text:
        formatted_text = text
        if not formatted_text[-1] in ['.', '!', '?']:
            formatted_text += '.'
        formatted_text = formatted_text[0].upper() + formatted_text[1:] if formatted_text and formatted_text[0].islower() else formatted_text
        manual_sentences.append(formatted_text)
    
    logger.info(f"Segmented text into {len(manual_sentences)} sentences using manual approach")
    return manual_sentences

def simple_break_long_sentence(long_text):
    """Simplified approach to break long sentences at natural pause points."""
    result = []
    
    # Try to break at clear pause points - conjunctions with capital letter following
    text = re.sub(r'(and|but|or|so)\s+([A-Z])', r'\1. \2', long_text)
    
    # Split at sentence endings
    parts = re.split(r'(?<=[.!?])\s+', text)
    
    current_part = ""
    word_count = 0
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        part_words = len(part.split())
        
        # If this part is already long enough to be its own sentence
        if part_words >= 10:
            # Add current accumulation if any
            if current_part:
                if not current_part[-1] in ['.', '!', '?']:
                    current_part += '.'
                current_part = current_part[0].upper() + current_part[1:] if current_part and current_part[0].islower() else current_part
                result.append(current_part)
                current_part = ""
                word_count = 0
            
            # Format this part
            if not part[-1] in ['.', '!', '?']:
                part += '.'
            part = part[0].upper() + part[1:] if part and part[0].islower() else part
            result.append(part)
        else:
            # If adding this would make current_part too long
            if word_count + part_words > 25:
                # Finish current part
                if current_part:
                    if not current_part[-1] in ['.', '!', '?']:
                        current_part += '.'
                    current_part = current_part[0].upper() + current_part[1:] if current_part and current_part[0].islower() else current_part
                    result.append(current_part)
                
                # Start new with this part
                current_part = part
                word_count = part_words
            else:
                # Add to current part
                if current_part:
                    current_part += " " + part
                else:
                    current_part = part
                word_count += part_words
    
    # Don't forget any remaining text
    if current_part:
        if not current_part[-1] in ['.', '!', '?']:
            current_part += '.'
        current_part = current_part[0].upper() + current_part[1:] if current_part and current_part[0].islower() else current_part
        result.append(current_part)
    
    return result


def ai_segment_transcription(text, max_chunk_size=5000):
    """
    Segment transcription into sentences using DeepSeek AI.
    Processes text in chunks to avoid timeouts with large transcriptions.
    
    Args:
        text (str): The transcription text to segment
        max_chunk_size (int): Maximum size of each chunk in characters
        
    Returns:
        list: List of segmented sentences
    """
    if not text or not text.strip():
        return []
    
    # If text is short enough, process it directly
    if len(text) <= max_chunk_size:
        try:
            return _process_text_with_deepseek(text)
        except Exception as e:
            logger.error(f"DeepSeek AI sentence segmentation failed: {e}")
            # Fallback to regex-based sentence splitting
            return simple_sentence_split(text)
    
    # For longer text, split into chunks with overlap to ensure sentence continuity
    chunks = []
    overlap = 100  # Characters to overlap between chunks
    start = 0
    
    logger.info(f"Text too large ({len(text)} chars), splitting into chunks of {max_chunk_size} chars")
    
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        
        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Try to find sentence-ending punctuation for a clean break
            break_point = find_break_point(text, max(start, end - overlap), end)
            chunk = text[start:break_point]
            start = break_point
        else:
            chunk = text[start:end]
            start = end
            
        chunks.append(chunk)
    
    logger.info(f"Split text into {len(chunks)} chunks for processing")
    
    # Process each chunk and combine results
    all_sentences = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
        try:
            sentences = _process_text_with_deepseek(chunk)
            logger.info(f"Chunk {i+1}: got {len(sentences)} sentences from DeepSeek")
            all_sentences.extend(sentences)
        except Exception as e:
            logger.error(f"DeepSeek AI chunk segmentation failed: {e}")
            sentences = simple_sentence_split(chunk)
            logger.info(f"Chunk {i+1}: used fallback to get {len(sentences)} sentences")
            all_sentences.extend(sentences)
    
    logger.info(f"Total sentences after processing all chunks: {len(all_sentences)}")
    return all_sentences

def _process_text_with_deepseek(text):
    """Process a single chunk of text with DeepSeek AI"""
    prompt = f"""
You are a transcription editor. Your job is to take raw, unpunctuated or poorly segmented transcript text and break it into clean, properly punctuated, natural-sounding sentences.

Here is the raw transcription text:

{text.strip()}

Please return the same text with:
- Accurate sentence boundaries
- Proper capitalization and punctuation
- Natural rhythm and flow of spoken English
- No added content, just clean formatting
- Do not show "Here's the cleaned up version"

Only return the corrected text.
""".strip()

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        },
        timeout=30
    )

    response.raise_for_status()
    result = response.json()
    ai_response = result["choices"][0]["message"]["content"].strip()
    return ai_response.split("\n\n")

def simple_sentence_split(text):
    """Simple regex-based sentence splitting as fallback"""
    # Split on common sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

def find_break_point(text, start, end):
    """Find a good break point (end of sentence) between start and end positions"""
    # Look for sentence-ending punctuation
    for i in range(end, start, -1):
        if i < len(text) and i > 0 and text[i-1] in '.!?' and (i == len(text) or text[i].isspace()):
            return i
    
    # If no good break point found, just use the end position
    return end

    

# Add this function for MP4 transcription after your pdf text extraction functions
def transcribe_mp4(video_file_path):
    """Extract audio from MP4 and convert it to text using speech recognition with chunking for large files"""
    try:
        # Create a temporary directory for the audio file
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        # Log file size
        file_size = os.path.getsize(video_file_path)
        logger.info(f"Processing MP4 file: {video_file_path}, size: {file_size/1024/1024:.2f} MB")
        
        # Extract audio from video
        logger.info(f"Extracting audio from {video_file_path}")
        
        try:
            # Create video clip
            video = mp.VideoFileClip(video_file_path)
            
            # Check if audio exists
            if video.audio is None:
                logger.warning(f"No audio stream found in {video_file_path}")
                return [{"page": 1, "text": "No audio detected in the video file."}]
            
            # Log duration
            logger.info(f"Video duration: {video.duration:.2f} seconds")
            
            # Write audio to file
            logger.info(f"Extracting audio to {audio_path}")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            logger.info(f"Audio extraction complete")
            
            # Log audio file size
            audio_size = os.path.getsize(audio_path)
            logger.info(f"Extracted audio size: {audio_size/1024/1024:.2f} MB")
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [{"page": 1, "text": f"Error extracting audio: {str(e)}"}]
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Load audio file to determine length
        try:
            audio = AudioSegment.from_file(audio_path)
            audio_duration = len(audio) / 1000  # Duration in seconds
            logger.info(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Determine if we need to use chunking
            use_chunking = audio_duration > 45  # Use chunking for files longer than 45 seconds
            
            if use_chunking:
                logger.info(f"Audio is long ({audio_duration:.2f}s), using chunking approach")
                
                # Set chunk size (30 seconds chunks with slight overlap)
                chunk_length_ms = 30 * 1000  # 30 seconds
                overlap_ms = 1000  # 1 second overlap to avoid cutting words
                
                # Calculate number of chunks
                num_chunks = math.ceil(len(audio) / (chunk_length_ms - overlap_ms))
                chunks = []
                
                # Create overlapping chunks
                for i in range(num_chunks):
                    start_ms = max(0, i * (chunk_length_ms - overlap_ms))
                    end_ms = min(len(audio), start_ms + chunk_length_ms)
                    chunk = audio[start_ms:end_ms]
                    chunks.append(chunk)
                
                logger.info(f"Split audio into {len(chunks)} chunks")
                
                # Process each chunk
                transcribed_chunks = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    # Export chunk to temporary file
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                    chunk.export(chunk_path, format="wav")
                    
                    # Transcribe the chunk with retry mechanism
                    chunk_text = ""
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            with sr.AudioFile(chunk_path) as source:
                                chunk_audio = recognizer.record(source)
                                chunk_text = recognizer.recognize_google(chunk_audio)
                                logger.info(f"Chunk {i+1} transcribed: {len(chunk_text)} chars")
                                break  # Success, exit retry loop
                        except sr.RequestError as e:
                            logger.error(f"API request error on chunk {i+1}, attempt {attempt+1}: {e}")
                            if attempt == max_retries - 1:  # Last attempt
                                chunk_text = f"[API error on segment {i+1}]"
                            time.sleep(1)  # Wait before retry
                        except sr.UnknownValueError:
                            logger.error(f"Could not understand audio in chunk {i+1}, attempt {attempt+1}")
                            if attempt == max_retries - 1:  # Last attempt
                                chunk_text = f"[Unintelligible audio in segment {i+1}]"
                            time.sleep(1)  # Wait before retry
                        except Exception as e:
                            logger.error(f"Error transcribing chunk {i+1}, attempt {attempt+1}: {e}")
                            if attempt == max_retries - 1:  # Last attempt
                                chunk_text = f"[Error in segment {i+1}]"
                            time.sleep(1)  # Wait before retry
                    
                    transcribed_chunks.append(chunk_text)
                    
                    # Clean up chunk file
                    try:
                        os.remove(chunk_path)
                    except Exception as e:
                        logger.error(f"Error removing chunk file: {e}")
                
                # Combine chunks into final transcription
                transcribed_text = " ".join(transcribed_chunks)
                logger.info(f"All chunks processed. Total transcription length: {len(transcribed_text)} chars")
                
            else:
                # For shorter audio, process in one go
                logger.info("Audio is short enough for single-pass transcription")
                transcribed_text = ""
                max_retries = 3
                
                for attempt in range(max_retries):
                    try:
                        with sr.AudioFile(audio_path) as source:
                            logger.info("Recording audio data...")
                            audio_data = recognizer.record(source)
                            logger.info("Sending to Google speech recognition...")
                            transcribed_text = recognizer.recognize_google(audio_data)
                            logger.info(f"Successfully transcribed audio: {len(transcribed_text)} chars")
                            break  # Success, exit retry loop
                    except sr.RequestError as e:
                        logger.error(f"API request error, attempt {attempt+1}: {e}")
                        if attempt == max_retries - 1:  # Last attempt
                            transcribed_text = f"Could not transcribe audio. API error: {str(e)}"
                        time.sleep(1)  # Wait before retry
                    except sr.UnknownValueError:
                        logger.error(f"Could not understand audio, attempt {attempt+1}")
                        if attempt == max_retries - 1:  # Last attempt
                            transcribed_text = "Speech recognition could not understand audio."
                        time.sleep(1)  # Wait before retry
                    except Exception as e:
                        logger.error(f"Speech recognition error, attempt {attempt+1}: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        if attempt == max_retries - 1:  # Last attempt
                            transcribed_text = f"Could not transcribe audio. Error: {str(e)}"
                        time.sleep(1)  # Wait before retry
            
        except Exception as audio_error:
            logger.error(f"Error processing audio: {audio_error}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            transcribed_text = f"Could not process audio. Error: {str(audio_error)}"
        
        # Apply sentence segmentation to the transcribed text
        try:
            sentences = ai_segment_transcription(transcribed_text)
            logger.info(f"Segmented transcription into {len(sentences)} sentences")
            
            # Format the transcribed text with proper sentences
            formatted_text = "\n\n".join(sentences)
            
            # Clean up
            video.close()
            if os.path.exists(audio_path):
                os.remove(audio_path)
            os.rmdir(temp_dir)
            
            # Return the formatted text as a page
            return [{"page": 1, "text": formatted_text}]
        except Exception as e:
            logger.error(f"Error segmenting transcription: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return the raw transcribed text if segmentation fails
            return [{"page": 1, "text": transcribed_text}]
        
    except Exception as e:
        logger.error(f"Error transcribing MP4: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Clean up any temporary files if possible
        try:
            if 'video' in locals() and video:
                video.close()
            if 'temp_dir' in locals() and temp_dir and os.path.exists(temp_dir):
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        
        # Return a placeholder result to allow processing to continue
        return [{"page": 1, "text": f"Error transcribing video file: {str(e)}"}]


    

def extract_text_from_docx(docx_file_path):
    """Extract text from a .docx file by page."""
    try:
        doc = Document(docx_file_path)
        pages = []
        page_text = ""
        page_count = 1
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                page_text += text + "\n"
                
                # You may need to adjust logic for determining page breaks
                # This is a simplified approach
                if "\f" in text:  # Form feed character (page break)
                    pages.append({"page": page_count, "text": page_text})
                    page_text = ""
                    page_count += 1
        
        # Add the last page if it has content
        if page_text:
            pages.append({"page": page_count, "text": page_text})
            
        return pages
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return []

def split_into_sentences(text):
    """Split text into individual sentences using spaCy with special handling for page breaks."""
    # First, pre-process to handle potential page break markers
    # Replace any common page break indicators with sentence-ending punctuation
    text = re.sub(r'(\w+)[\s]*\f[\s]*(\w+)', r'\1. \2', text)  # Form feed character
    text = re.sub(r'(\w+)[\s]*\[PAGE\s*\d+\][\s]*(\w+)', r'\1. \2', text)  # [PAGE X] marker
    
    # Process with spaCy
    doc = nlp(text)
    
    # Extract sentences, ensuring each is properly bounded
    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:  # Only add non-empty sentences
            sentences.append(sent_text)
    
    # Post-process: check for obvious split sentences
    processed_sentences = []
    buffer = ""
    
    for sentence in sentences:
        # If previous buffer exists and current sentence starts lowercase or with a continuation marker
        if buffer and (sentence[0].islower() or sentence.startswith(('and', 'or', 'but', 'so', 'because', 'which', 'that'))):
            buffer += " " + sentence
        else:
            # Save previous buffer if any
            if buffer:
                processed_sentences.append(buffer)
            buffer = sentence
    
    # Add the last buffer
    if buffer:
        processed_sentences.append(buffer)
    
    return processed_sentences


def call_deepseek_api(prompt):
    """Call Deepseek API with token usage tracking"""
    try:
        # Calculate approximate input tokens
        input_tokens = len(prompt.split())
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        payload = {
            "model": "deepseek-chat",  # adjust based on Deepseek model names
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        start_time = time.time()
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response_time = time.time() - start_time
        
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        
        # Extract token usage if available
        usage = result.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', input_tokens)  # Fallback to estimate
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        # Log token usage for cost calculation
        logger.info(f"Deepseek API call - Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        logger.info(f"Response time: {response_time:.2f} seconds")
        
        # Calculate estimated cost (adjust pricing as needed)
        prompt_cost = (prompt_tokens / 1000) * 0.002  # Example rate: $0.002 per 1K tokens
        completion_cost = (completion_tokens / 1000) * 0.006  # Example rate: $0.006 per 1K tokens
        total_cost = prompt_cost + completion_cost
        
        logger.info(f"Estimated cost: ${total_cost:.6f} (Prompt: ${prompt_cost:.6f}, Completion: ${completion_cost:.6f})")
        
        # Store token usage for session totals
        global session_token_usage
        if 'session_token_usage' not in globals():
            session_token_usage = {
                'prompt_tokens': 0, 
                'completion_tokens': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'api_calls': 0
            }
        
        session_token_usage['prompt_tokens'] += prompt_tokens
        session_token_usage['completion_tokens'] += completion_tokens
        session_token_usage['total_tokens'] += total_tokens
        session_token_usage['total_cost'] += total_cost
        session_token_usage['api_calls'] += 1
        
        # Log cumulative session usage periodically
        if session_token_usage['api_calls'] % 5 == 0:  # Log every 5 calls
            logger.info(f"===== SESSION USAGE SUMMARY =====")
            logger.info(f"Total API calls: {session_token_usage['api_calls']}")
            logger.info(f"Total tokens: {session_token_usage['total_tokens']} (Prompt: {session_token_usage['prompt_tokens']}, Completion: {session_token_usage['completion_tokens']})")
            logger.info(f"Estimated total cost: ${session_token_usage['total_cost']:.6f}")
            logger.info(f"================================")
        
        # Adjust the parsing based on Deepseek's response format
        return result.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    except Exception as e:
        logger.error(f"Error calling Deepseek API: {e}")
        return None

    
    
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




# Function to check financial rules that require confirmation of current year regulations

def check_financial_rule_verification(content):
    financial_issues = []

    # Terms related to financial rules that require date-specific confirmation
    date_sensitive_terms = ["rmd", "required minimum distribution", "contribution", "distribution", 
                            "income limit", "catch-up contribution", "rule change", "2024", "2023"]
    # If any date-sensitive terms are found in the content, add a verification reminder
    for term in date_sensitive_terms:
        if term.lower() in content.lower():
            financial_issues.append(

                "Be sure to confirm the accuracy of all statements made relating to current year financial rules."
            )
            
            break  # Only add the statement once
    return financial_issues


# Function to remove quotes from the analyzed text

def remove_quotes(text):
    """
    Remove all types of quotes (single, double) from the given text.
    """
    return re.sub(r'[\'"]', '', text)


# Function for Prohibited Words display

def generate_clean_finra_response(prompt, prohibited_words_dict, max_attempts=3):
    """
    Keeps calling Claude until the response has no prohibited words
    in the 'Specific Compliant Alternative' lines (or until max_attempts).
    Returns the final cleaned FINRA response or None if not possible.
    """

    # Build the system message once
    system_message = f"""
You are a compliance assistant specializing in financial regulations.
Do NOT use these words: {', '.join(prohibited_words_dict.keys())}
"""

    for _ in range(max_attempts):
        # Call Claude only once per loop
        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            system=system_message,  # system message as separate parameter
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        finra_response = response.content[0].text.strip()

        # 2) Find "Specific Compliant Alternative" lines
        matches = re.findall(
            r'Specific Compliant Alternative:\s*"([^"]*)"',
            finra_response,
            flags=re.IGNORECASE
        )
        
        # 3) Check each found alternative for prohibited words
        has_prohibited = False
        for alt_text in matches:
            for pword in prohibited_words_dict.keys():
                pattern = rf"\b{re.escape(pword.lower())}\b"
                if re.search(pattern, alt_text.lower()):
                    has_prohibited = True
                    break
            if has_prohibited:
                break
        if not has_prohibited:
            # Found a version with no banned words
            return finra_response

    # If we got here, can't find a clean alternative after max_attempts
    return None

# Function to check if text is inside quotes in the original text
def is_inside_quotes(text, original_text):
    """Check if the text appears inside quotes in the original text."""
    # Find all quoted text
    quoted_texts = re.findall(r'"([^"]*)"', original_text)
    quoted_texts += re.findall(r"'([^']*)'", original_text)
    
    # Check if text appears in any of the quoted sections
    for quoted_text in quoted_texts:
        if text in quoted_text:
            return True
    return False

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
    """Checks text compliance using BERT for flagging and GPT for verification."""
    try:
        # Get the BERT model and tokenizer - this will load it if not already loaded
        global BERT_MODEL, BERT_TOKENIZER
        BERT_MODEL, BERT_TOKENIZER = get_bert_model()

        # If text is empty, contains only whitespace, or is just a bullet point
        if not text or text.isspace() or text == "•" or text == "\u2022":
            logger.info("Text is empty or just a bullet point - skipping")
            return {"compliant": True, "flagged_instances": []}

        # If text is a dictionary, extract the 'text' field
        if isinstance(text, dict):
            text = text.get('text', '')
            logger.info(f"Extracted text from dictionary, length: {len(text)}")
        
        # Clean the text
        original_len = len(text)
        text = re.sub(r'[•\u2022]', '', text)  # Remove bullet points
        text = re.sub(r'\s+', ' ', text).strip()  # Clean up whitespace
        logger.info(f"Cleaned text: {original_len} chars → {len(text)} chars")
        
        # Split text into sentences
        logger.info(f"Splitting text into sentences...")
        sentences = split_into_sentences(text)
        logger.info(f"Found {len(sentences)} sentences")
        flagged_instances = []
        bert_flagged_instances = []

        # Process sentences in batches
        batch_size = 1024  # Increased from 512
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(sentences)-1)//batch_size + 1}: {len(batch)} sentences")
        
            for sentence_idx, sentence in enumerate(batch):
                # Skip invalid sentences
                if not sentence.strip() or sentence.strip() in ['•', '\u2022']:
                    continue
            
                # Clean the sentence
                sentence = re.sub(r'[•\u2022]', '', sentence).strip()
                if not sentence:  # Skip if sentence becomes empty after cleaning
                    continue
                
                # Log only occasional sentences to avoid overwhelming logs
                if sentence_idx % 20 == 0:  # Log every 20th sentence
                    logger.info(f"Processing sentence #{i + sentence_idx}: {sentence[:50]}...")
                
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

                # Add explicit log for each sentence prediction
                logger.info(f"Prediction: {prediction} ('{'non-compliant' if prediction == 1 else 'compliant'}'), Confidence: {confidence:.1%}")

                # Log predictions only for flagged or every 20th sentence
                if prediction == 1 or sentence_idx % 20 == 0:
                    logger.info(f"BERT prediction for sentence #{i + sentence_idx}: {prediction} ('{'non-compliant' if prediction == 1 else 'compliant'}'), Confidence: {confidence:.1%}")

                if prediction == 1 and len(sentence.split()) > 2:  # Non-compliant and meaningful length
                    # Store this as a potential flag from BERT
                    bert_flagged_instances.append({
                        "flagged_instance": sentence,
                        "confidence": confidence,
                        "page": page_num
                    })
                    logger.info(f"BERT flagged sentence (confidence: {confidence:.1%}): {sentence[:100]}...")

        # Log summary of BERT analysis
        logger.info(f"BERT analysis complete: {len(sentences)} total sentences, {len(bert_flagged_instances)} flagged instances")
        
        # Use GPT to verify BERT's findings
        if bert_flagged_instances:
            logger.info(f"Sending {len(bert_flagged_instances)} flagged instances to GPT for verification...")
            verified_instances = verify_with_gpt(bert_flagged_instances)
            flagged_instances = verified_instances
            logger.info(f"GPT verification complete: {len(verified_instances)} of {len(bert_flagged_instances)} confirmed as non-compliant")
        else:
            logger.info("No instances flagged by BERT, skipping GPT verification")

        # Filter out any remaining invalid instances
        original_count = len(flagged_instances)
        flagged_instances = [
            instance for instance in flagged_instances
            if instance.get("flagged_instance") 
            and len(instance["flagged_instance"].strip()) > 5
            and instance["flagged_instance"].strip() not in ['•', '\u2022']
        ]
        if original_count != len(flagged_instances):
            logger.info(f"Filtered out {original_count - len(flagged_instances)} invalid instances")

        logger.info(f"Final compliance check results:")
        logger.info(f"  - Total sentences processed: {len(sentences)}")
        logger.info(f"  - BERT flagged instances: {len(bert_flagged_instances)}")
        logger.info(f"  - Final flagged instances after GPT verification: {len(flagged_instances)}")
        logger.info(f"  - Overall compliance status: {'Compliant' if len(flagged_instances) == 0 else 'Non-Compliant'}")
        
        return {
            "compliant": len(flagged_instances) == 0,
            "flagged_instances": flagged_instances
        }
    
    except Exception as e:
        logger.error(f"Error during compliance check: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"compliant": False, "error": f"An error occurred during compliance checking: {str(e)}"}

    

def verify_with_gpt(bert_flagged_instances):
    """Verifies BERT-flagged instances using Deepseek to reduce false positives."""
    verified_instances = []
    
    logger.info(f"========== GPT VERIFICATION STARTED ==========")
    logger.info(f"Verifying {len(bert_flagged_instances)} instances flagged by BERT")
    
    # Initialize batch metrics
    batch_tokens = 0
    batch_cost = 0.0
    
    for i, instance in enumerate(bert_flagged_instances, 1):
        sentence = instance.get("flagged_instance", "")
        page_num = instance.get("page")
        confidence = instance.get("confidence", 0)
        
        logger.info(f"------- Instance {i}/{len(bert_flagged_instances)} -------")
        logger.info(f"Page: {page_num}, Confidence: {confidence:.1%}")
        logger.info(f"Text: \"{sentence[:100]}{'...' if len(sentence) > 100 else ''}\"")
        
        try:
            logger.info(f"Verifying with GPT if instance is non-compliant...")
            verification_prompt = f"""
Determine if this text violates FINRA's communication rules by being false, misleading, promissory or exaggerated:

"{sentence}"

Answer with ONLY "YES" or "NO", followed by a brief explanation.
"YES" means it IS non-compliant.
"NO" means it IS compliant.

IMPORTANT DISTINCTION:
- Non-compliant: Statements that present specific financial benefits, tax advantages, or performance outcomes as definite facts without proper qualifiers
- Compliant: General statements, subjective opinions, or descriptions that don't make specific claims about financial outcomes or benefits

CRITICAL: Statements presented as definitive facts without qualifying language are typically non-compliant when they involve:
- Tax benefits
- Investment outcomes
- Financial advantages
- Product features that don't universally apply

Examples of non-compliant statements:
- "A Traditional IRA is a place to put your money to save on taxes." (presents tax saving as definite)
- "Roth IRAs are a great vehicle for tax free investing" (presents absolute statement when additioal rules apply to receive benefits of investing in ROTH IRA)
- "IRAs are vehicles with tax advantages" (not necessarily true in all cases)
- "This fund outperforms the market" (absolute claim without qualification)
- "The strategy protects your assets during downturns" (unqualified protection claim)

Examples of compliant alternatives:
- "A Traditional IRA is a place to put your money to potentially save on taxes."
- "Roth IRAs may offer tax advantages for qualifying investors" or "Roth IRAs are a potential vehicle for tax advantaged investing"
- "IRAs are vehicles with potential tax advantages" (clarifies all advantages don't apply to everyone)
- "This fund is designed to seek competitive returns relative to its benchmark"
- "The strategy aims to help manage risk during market downturns"

Always answer "YES" (non-compliant) for statements that:
1. Present possible, implied or conditional benefits as definite outcomes
2. Make absolute claims about tax advantages
3. Lack qualifiers like "may," "potential," "designed to," or "aims to" when discussing benefits
4. State as fact something that doesn't apply in all circumstances

All financial benefits and advantages MUST be qualified with appropriate language.
"""
            
            
            verification_response = call_deepseek_api(verification_prompt)
            logger.info(f"GPT verification response: {verification_response}")
            
            # Track token usage from global session data
            if 'session_token_usage' in globals():
                batch_tokens = session_token_usage['total_tokens']
                batch_cost = session_token_usage['total_cost']
            
            # More strict evaluation of GPT's response
            is_non_compliant = False
            if verification_response:
                # Check if the response starts with YES
                first_word = verification_response.strip().split()[0].upper() if verification_response.strip() else ""
                is_non_compliant = first_word == "YES"
                
                # For high-confidence BERT predictions, verify we're not getting a false negative
                if not is_non_compliant and confidence > 0.9:
                    # Look for known violation patterns in the text
                    violation_patterns = [
                        r'\bwill\s+(?:provide|earn|gain|make|increase|grow|guarantee)',
                        r'\bguarantee[ds]?\b',
                        r'\balways\b',
                        r'\bnever\b',
                        r'\bcertain(?:ly)?\b',
                        r'\bensure[ds]?\b'
                    ]
                    
                    for pattern in violation_patterns:
                        if re.search(pattern, sentence.lower()):
                            is_non_compliant = True
                            logger.info(f"Overriding GPT decision due to high confidence and violation pattern match")
                            break
            
            if is_non_compliant:
                logger.info(f"GPT CONFIRMS non-compliance, generating alternative...")
                
                # Simplified prompt to get a direct response
                alternative_prompt = f"""
                This text violates FINRA rules: "{sentence}"
                
                Rewrite it to be compliant by:
                1. Replacing overstatements
                2. Removing absolute statements
                3. Adding appropriate qualifiers
                4. Avoiding guarantees or promises
                5. Maintaining the original meaning
                
                Respond with ONLY the compliant alternative text and nothing else.
                """
                
                # Get direct alternative without needing complex parsing
                compliant_text = call_deepseek_api(alternative_prompt).strip()
                logger.info(f"Direct alternative received: {compliant_text}")
                
                # Validate the response - should be a simple text string
                if not compliant_text or len(compliant_text) < 5:
                    logger.error("Got empty or too short alternative, using API fallback")
                    # Make one more attempt with a more explicit instruction
                    fallback_prompt = f"""
                    Rewrite this non-compliant financial text: "{sentence}"
                    Make it FINRA-compliant by removing any guarantees.
                    Return ONLY the rewritten text with no other text, explanations, or formatting.
                    """
                    compliant_text = call_deepseek_api(fallback_prompt).strip()
                
                # Clean quotes if present
                if compliant_text.startswith('"') and compliant_text.endswith('"'):
                    compliant_text = compliant_text[1:-1]
                
                # Use a simple standard rationale
                rationale = "This text may contain absolutes, guarantees, or promissory statements not allowed under FINRA regulations."
                
                instance = {
                    "flagged_instance": sentence,
                    "compliance_status": "non-compliant",
                    "specific_compliant_alternative": compliant_text,
                    "rationale": rationale,
                    "page": page_num,
                    "confidence": f"{confidence:.1%}",
                    "gpt_verified": True
                }
                verified_instances.append(instance)
                logger.info(f"✓ GPT verified instance as non-compliant and added to results")
            else:
                logger.info(f"✗ GPT determined instance is COMPLIANT - FALSE POSITIVE from BERT - not flagging")
                
        except Exception as e:
            logger.error(f"Error verifying with GPT: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # If verification fails, fall back to BERT judgment for high-confidence predictions
            if confidence > 0.8:
                logger.warning(f"GPT verification failed, using BERT judgment for high confidence ({confidence:.1%})")
                # Direct API call for alternative
                fallback_prompt = f"""
                Rewrite this text to be FINRA-compliant: "{sentence}"
                Return ONLY the rewritten text.
                """
                fallback_alternative = call_deepseek_api(fallback_prompt).strip()
                
                if not fallback_alternative or len(fallback_alternative) < 5:
                    fallback_alternative = "We may help clients pursue their investment objectives while understanding that all investments involve risk."
                
                instance = {
                    "flagged_instance": sentence,
                    "compliance_status": "non-compliant",
                    "specific_compliant_alternative": fallback_alternative,
                    "rationale": "This text may not comply with FINRA regulations (GPT verification failed).",
                    "page": page_num,
                    "confidence": f"{confidence:.1%}",
                    "gpt_verified": False
                }
                verified_instances.append(instance)
                logger.info(f"Added instance due to verification failure but high BERT confidence")
    
    # Calculate final token usage and costs for this verification batch
    batch_tokens_used = 0
    batch_final_cost = 0.0
    
    if 'session_token_usage' in globals():
        batch_tokens_used = session_token_usage['total_tokens'] - batch_tokens
        batch_final_cost = session_token_usage['total_cost'] - batch_cost
    
    logger.info(f"========== GPT VERIFICATION COMPLETED ==========")
    logger.info(f"Results: {len(bert_flagged_instances)} BERT flags → {len(verified_instances)} verified flags")
    logger.info(f"Tokens used in this verification batch: {batch_tokens_used}")
    logger.info(f"Estimated cost for this batch: ${batch_final_cost:.6f}")
    
    return verified_instances


def generate_fallback_alternative(non_compliant_text):
    """
    Generate a simple fallback alternative when GPT parsing fails,
    without using hardcoded pattern replacements.
    """
    try:
        # Try to get a compliant alternative using another API call
        prompt = f"""
        Convert this potentially non-compliant financial text into a FINRA-compliant version:
        
        "{non_compliant_text}"
        
        Make it compliant by:
        1. Removing absolute statements
        2. Adding appropriate qualifiers
        3. Avoiding guarantees
        4. Maintaining the original meaning as much as possible
        
        Return ONLY the compliant alternative text with no extra explanation.
        """
        
        compliant_version = call_deepseek_api(prompt).strip()
        
        # Check if we got a valid response
        if compliant_version and len(compliant_version) > 10:
            logger.info(f"Generated fallback using additional API call: {compliant_version[:100]}...")
            return compliant_version
    except Exception as e:
        logger.error(f"Error generating fallback with API: {e}")
    
    # If the API call fails or returns an invalid response, use a generic but still useful alternative
    logger.info("Using generic fallback text")
    return "We strive to help clients pursue their investment objectives while understanding that all investments involve risk."




def get_compliant_alternative(non_compliant_text):
    """Gets compliant alternative from database."""
    try:
        conn = psycopg2.connect(
            dbname="postgresql_instance_free",
            user="postgresql_instance_free_user",
            password="bz3SdnKi6g6TRdM4j1AtE2Ash8VNgiQO",
            host="dpg-cts4psa3esus73dn1cn0-a.oregon-postgres.render.com",
            port="5432"
        )
        cursor = conn.cursor()
        
        # Try to find exact match first
        cursor.execute("""
            SELECT compliant 
            FROM compliance_examples 
            WHERE non_compliant = %s
        """, (non_compliant_text,))
        
        result = cursor.fetchone()
        
        # If no exact match, try to find similar text
        if not result:
            cursor.execute("""
                SELECT compliant, non_compliant
                FROM compliance_examples
                ORDER BY similarity(non_compliant, %s) DESC
                LIMIT 1
            """, (non_compliant_text,))
            result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return result[0] if result else "Review and revise this content to ensure compliance with regulations."
        
    except Exception as e:
        logger.error(f"Error getting compliant alternative: {e}")
        return "Review and revise this content to ensure compliance with regulations."

    
def reconcile_compliance_checks(first_check, second_check):
    """
    Reconcile results from both compliance checks.
    Takes the more conservative approach when there are differences and includes all unique instances.
    """
    reconciled = {
        "compliant": True,
        "flagged_instances": [],
        "message": "",
        "additional_findings": []  # New field to track instances found only in second check
    }

    # Combine flagged instances from both checks
    all_instances = {}
    second_check_keys = set()
    
    # Process first check instances
    for instance in first_check.get('flagged_instances', []):
        key = instance.get('flagged_text', '').lower()
        all_instances[key] = {
            **instance,
            "found_in": ["first_check"]
        }

    # Process second check instances
    for instance in second_check.get('flagged_instances', []):
        key = instance.get('flagged_text', '').lower()
        second_check_keys.add(key)
        
        if key in all_instances:
            # Update existing instance
            all_instances[key]["found_in"].append("second_check")
            
            # Take the more conservative status
            current_status = all_instances[key].get('compliance_status', '').lower()
            new_status = instance.get('compliance_status', '').lower()
            if new_status == 'non-compliant' or current_status == 'non-compliant':
                all_instances[key]['compliance_status'] = 'non-compliant'
            elif new_status == 'partially compliant' or current_status == 'partially compliant':
                all_instances[key]['compliance_status'] = 'partially compliant'
            
            # Merge rationales if different
            if instance.get('rationale') and instance['rationale'] != all_instances[key].get('rationale'):
                all_instances[key]['rationale'] = f"First check: {all_instances[key].get('rationale', '')} | Second check: {instance['rationale']}"
        else:
            # New instance found only in second check
            instance['found_in'] = ["second_check"]
            all_instances[key] = instance
            reconciled['additional_findings'].append({
                'flagged_text': instance.get('flagged_text'),
                'compliance_status': instance.get('compliance_status'),
                'specific_compliant_alternative': instance.get('specific_compliant_alternative'),
                'rationale': instance.get('rationale')
            })

    # Convert to list and sort by compliance status severity
    def severity_score(status):
        status = status.lower()
        scores = {
            'non-compliant': 3,
            'partially compliant': 2,
            'more compliant': 1,
            'compliant': 0
        }
        return scores.get(status, 0)

    reconciled['flagged_instances'] = sorted(
        all_instances.values(),
        key=lambda x: severity_score(x.get('compliance_status', '')),
        reverse=True
    )

    # Add verification summary
    reconciled['verification_summary'] = {
        'total_unique_flags': len(all_instances),
        'flags_in_first_check': len(first_check.get('flagged_instances', [])),
        'flags_in_second_check': len(second_check.get('flagged_instances', [])),
        'additional_flags_from_second_check': len(reconciled['additional_findings']),
        'flags_found_in_both_checks': len([i for i in all_instances.values() if len(i['found_in']) > 1])
    }

    # Combine messages
    messages = []
    # Instead of using both checks' general messages, create one concise message with specific suggestions
    if reconciled['flagged_instances']:
        main_message = "This text could be improved in the following ways:"
        suggestions = []
        for instance in reconciled['flagged_instances']:
            if instance.get('specific_compliant_alternative'):
                suggestions.append(instance.get('specific_compliant_alternative'))
        
        reconciled['message'] = f"{main_message} {' '.join(suggestions)}"
    else:
        reconciled['message'] = "The text appears to be compliant with FINRA's communication rules."
    
    return reconciled

# Set the path for ffmpeg in pydub
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"


# APP = FLASK(__name__)
app = Flask(__name__)

def create_pdf_from_text(text, output_path):
    """Create a PDF file from transcribed text with better formatting"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY
    
    # Create a custom style for the transcription text
    doc = SimpleDocTemplate(output_path, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', 
                              fontName='Helvetica',
                              fontSize=12,
                              leading=14,
                              alignment=TA_JUSTIFY,
                              spaceAfter=10))
    
    story = []
    
    # Add a title
    story.append(Paragraph("Transcription", styles['Heading1']))
    story.append(Spacer(1, 0.25*inch))
    
    # Add date and time of transcription
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {timestamp}", styles['Italic']))
    story.append(Spacer(1, 0.25*inch))
    
    # Split text into paragraphs - respect both double newlines and sentence endings
    parts = re.split(r'\n\n|\r\n\r\n', text)
    
    for part in parts:
        if not part.strip():
            continue
            
        # Further split long paragraphs by sentences for better readability
        if len(part) > 300:  # If paragraph is very long
            sentences = re.split(r'(?<=[.!?])\s+', part)
            current_paragraph = ""
            
            for sentence in sentences:
                if len(current_paragraph) + len(sentence) > 300:
                    # Add the current paragraph to the story
                    if current_paragraph:
                        story.append(Paragraph(current_paragraph, styles['Justify']))
                    current_paragraph = sentence
                else:
                    if current_paragraph:
                        current_paragraph += " " + sentence
                    else:
                        current_paragraph = sentence
            
            # Add any remaining text
            if current_paragraph:
                story.append(Paragraph(current_paragraph, styles['Justify']))
        else:
            # Add regular paragraph
            story.append(Paragraph(part, styles['Justify']))
        
        # Add space after each paragraph
        story.append(Spacer(1, 0.1*inch))
    
    # Build the PDF
    doc.build(story)
    logger.info(f"Created formatted PDF from transcribed text at: {output_path}")


@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename or not filename.endswith('.mp4'):
        return jsonify({'error': 'Invalid or missing MP4 filename'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Transcribe the video file
        logger.info(f"Processing video file: {filename}")
        text_by_page = transcribe_mp4(file_path)
        all_text = " ".join([page["text"] for page in text_by_page])
        
        # Create a PDF from the transcription
        base_name = os.path.splitext(filename)[0]
        pdf_filename = f"{base_name}_transcription.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        
        # Create PDF from transcribed text
        create_pdf_from_text(all_text, pdf_path)
        
        # Store the transcription results in memory for access as-is
        processed_files[filename] = {
            'text': all_text,
            'text_by_page': text_by_page,
            'revised_text': "",
            'disclosures': DISCLOSURE_WORDS,
            'sliced_disclosures': list(DISCLOSURE_WORDS.values())[:5] if DISCLOSURE_WORDS else [],
            'finra_analysis': [],
            'results': len(split_into_sentences(all_text))
        }
        
        # Mark processing as complete
        final_check_status[filename] = True
        logger.info(f"MP4 transcription completed for {filename}")
        
        # Now process the PDF through the existing pipeline by redirecting
        return jsonify({
            'success': True,
            'message': 'Video processed and converted to PDF for analysis',
            'redirect': f'/process/{pdf_filename}'  # Redirect to process the PDF instead
        })
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Error processing video: {str(e)}"
        }), 500

    

@app.route('/verification_only', methods=['POST'])
def verification_only():
    data = request.get_json()
    custom_text = data.get("text", "").strip()
    
    if not custom_text:
        logger.error("No text provided for quick verification check.")
        return jsonify({"error": "No text provided."}), 400
    
    try:
        # Split text into sentences for analysis
        sentences = split_into_sentences(custom_text)
        
        # Just check the first substantial sentence
        for sentence in sentences:
            # Skip very short or empty sentences
            if not sentence.strip() or len(sentence.split()) < 3:
                continue
                
            # Verify with Deepseek if sentence is truly non-compliant
            verification_prompt = f"""
Determine if this text violates FINRA's communication rules by being false, misleading, promissory or exaggerated:

"{sentence}"

Answer with ONLY "YES" or "NO", followed by a brief explanation.
"YES" means it IS non-compliant.
"NO" means it IS compliant.

IMPORTANT DISTINCTION:
- Non-compliant: Statements that present specific financial benefits, tax advantages, or performance outcomes as definite facts without proper qualifiers
- Compliant: General statements, subjective opinions, or descriptions that don't make specific claims about financial outcomes or benefits

CRITICAL: Statements presented as definitive facts without qualifying language are typically non-compliant when they involve:
- Tax benefits
- Investment outcomes
- Financial advantages
- Product features that don't universally apply

Examples of non-compliant statements:
- "A Traditional IRA is a place to put your money to save on taxes." (presents tax saving as definite)
- "Roth IRAs are a great vehicle for tax free investing" (presents absolute statement when additioal rules apply to receive benefits of investing in ROTH IRA)
- "IRAs are vehicles with tax advantages" (not necessarily true in all cases)
- "This fund outperforms the market" (absolute claim without qualification)
- "The strategy protects your assets during downturns" (unqualified protection claim)

Examples of compliant alternatives:
- "A Traditional IRA is a place to put your money to potentially save on taxes."
- "Roth IRAs may offer tax advantages for qualifying investors" or "Roth IRAs are a potential vehicle for tax advantaged investing"
- "IRAs are vehicles with potential tax advantages" (clarifies all advantages don't apply to everyone)
- "This fund is designed to seek competitive returns relative to its benchmark"
- "The strategy aims to help manage risk during market downturns"

Always answer "YES" (non-compliant) for statements that:
1. Present possible, implied or conditional benefits as definite outcomes
2. Make absolute claims about tax advantages
3. Lack qualifiers like "may," "potential," "designed to," or "aims to" when discussing benefits
4. State as fact something that doesn't apply in all circumstances

All financial benefits and advantages MUST be qualified with appropriate language.
"""
            
            verification_response = call_deepseek_api(verification_prompt)
            logger.info(f"Deepseek verification response: {verification_response}")
            
            # Check if Deepseek confirms this is non-compliant
            is_non_compliant = False
            first_word = ""
            if verification_response:
                first_word = verification_response.strip().split()[0].upper() if verification_response.strip() else ""
                is_non_compliant = first_word == "YES"
            
            # If we found a violation, return immediately
            if is_non_compliant:
                return jsonify({
                    "verification_response": verification_response,
                    "is_non_compliant": True
                }), 200
            
            # Only check the first substantial sentence
            break
            
        # If we got here, no immediate violations were found
        return jsonify({
            "verification_response": "NO - No immediate violations detected.",
            "is_non_compliant": False
        }), 200
        
    except Exception as e:
        logger.error(f"Error during quick verification: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "An error occurred during quick verification check.",
            "message": str(e)
        }), 500



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
    """Send password reset email using SendGrid"""
    try:
        # Configure message
        message = Mail(
            from_email='riley.r.giauque@gmail.com',  # Must be verified in SendGrid
            to_emails=email,
            subject='Password Reset Request',
            html_content=f'''
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px;">
                    <h2 style="color: #004e98;">Password Reset Request</h2>
                    <p>You have requested to reset your password. Click the link below to set a new password:</p>
                    <p><a href="{reset_link}" style="display: inline-block; padding: 10px 20px; background-color: #00aaff; color: #ffffff; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
                    <p>If you did not request this reset, please ignore this email.</p>
                    <p>This link will expire in 24 hours.</p>
                </div>
            '''
        )
        
        # Send email
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        
        logger.info(f"Reset email sent to {email}, status code: {response.status_code}")
        return True
    except Exception as e:
        logger.error(f"Error sending password reset email: {e}")
        return False
    
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







# Sync Subscription Stautus with User in Stripe 
def sync_subscription_status(email):
    users = load_users()
    user = next((u for u in users if u['email'] == email), None)
    if not user:
        return {"error": "User not found in local database."}

    # 1) Query Stripe for the customer's active subscription
    customers = stripe.Customer.list(email=email).data
    if not customers:
        user['Subscribed'] = "No"
        user['Unsubscribed'] = "Yes"
        user['AccessEndDate'] = "Unknown"
        user['Sign Up Status'] = "Not Yet"  # No Stripe customer => certainly "Not Yet"
    else:
        customer = customers[0]
        subs = stripe.Subscription.list(customer=customer.id, status="active").data
        if subs:
            subscription = subs[0]

            # ─── [ NEW CODE ] ─────────────────────────────────────────
            # Expand this subscription to get its latest invoice
            full_sub = stripe.Subscription.retrieve(
                subscription.id, expand=["latest_invoice"]
            )
            # If there's an invoice and it's "paid," user has had at least 1 successful billing
            if full_sub.latest_invoice and full_sub.latest_invoice.status == "paid":
                user["Sign Up Status"] = "Signed Up"
            else:
                user["Sign Up Status"] = "Not Yet"
            # ─────────────────────────────────────────────────────────
            
            # Here's the new logic:
            if subscription.cancel_at_period_end:
                # They have scheduled a cancel; we consider them "unsubscribed" now
                user['Subscribed'] = "No"
                user['Unsubscribed'] = "Yes"
            else:
                # They do NOT have a pending cancel
                user['Subscribed'] = "Yes"
                user['Unsubscribed'] = "No"

            # Always set AccessEndDate from current_period_end
            next_billing_date = subscription.current_period_end
            user['AccessEndDate'] = datetime.utcfromtimestamp(next_billing_date).strftime('%Y-%m-%d')
        else:
            # No active sub
            user['Subscribed'] = "No"
            user['Unsubscribed'] = "Yes"
            user['AccessEndDate'] = "Unknown"
            user['Sign Up Status'] = "Not Yet"

    save_users(users)
    return user




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

# PROHIBITED WORDS FUNCTIONALITY

# Function for alligning prohibited words Control Panel with Results
@app.route('/get_prohibited_words', methods=['GET'])
def get_prohibited_words():

    try:
        conn = psycopg2.connect(
            dbname="postgresql_instance_free",
            user="postgresql_instance_free_user",
            password="bz3SdnKi6g6TRdM4j1AtE2Ash8VNgiQO",
            host="dpg-cts4psa3esus73dn1cn0-a.oregon-postgres.render.com",
            port="5432"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT word, alternative FROM prohibited_words;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Make a dict: { "word1": "alternative1", "word2": "alternative2", ... }
        prohibited_words_dict = {row[0]: row[1] for row in rows}
        return jsonify(prohibited_words_dict), 200
    except Exception as e:
        logger.info(f"Error fetching prohibited words from DB: {e}")
        return jsonify({"error": str(e)}), 500

# File to store prohibited words
STORAGE_FILE = 'prohibited_words.json'

# Function to load data from JSON file
def load_prohibited_words():
    try:
        with open(STORAGE_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Function to save data to JSON file
def save_prohibited_words(data):
    with open(STORAGE_FILE, 'w') as file:
        json.dump(data, file, indent=4)

# Function to delete prohibited word in control panel
@app.route('/delete_prohibited_word', methods=['POST'])
def delete_prohibited_word():
    data = request.get_json()
    word = data.get('word')
    if not word:
        return jsonify({'success': False, 'error': 'Word not provided'}), 400

    # Connect to DB
    try:
        conn = psycopg2.connect(
            dbname="postgresql_instance_free",
            user="postgresql_instance_free_user",
            password="bz3SdnKi6g6TRdM4j1AtE2Ash8VNgiQO",
            host="dpg-cts4psa3esus73dn1cn0-a.oregon-postgres.render.com",
            port="5432"
        )
        cursor = conn.cursor()

        # Delete the row where 'word' matches
        cursor.execute(
            "DELETE FROM prohibited_words WHERE word = %s",
            (word.lower(),)
        )
        rows_deleted = cursor.rowcount  # how many rows were actually deleted
        conn.commit()
        cursor.close()
        conn.close()

        if rows_deleted > 0:
            return jsonify({
                'success': True,
                'message': f'Prohibited word "{word}" deleted successfully from DB.'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Word "{word}" not found in DB.'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error deleting word from DB: {str(e)}'
        }), 500

# Function to add prohibited word
@app.route('/add_prohibited_word', methods=['POST'])
def add_prohibited_word():
    data = request.get_json()
    word = data.get('word', '').strip()
    alternative = data.get('alternative', '').strip()

    if not word:
        return jsonify({'success': False, 'error': 'Word must be provided.'}), 400


    # 1) Keep your JSON functionality
    # Load the current prohibited words from your JSON
    prohibited_words = load_prohibited_words()  

    # Update the dictionary in memory
    prohibited_words[word] = alternative if alternative else "No alternative provided"

    # Save the updated dictionary back to JSON
    save_prohibited_words(prohibited_words)     

    # 2) Also insert into PostgreSQL
    try:
        conn = psycopg2.connect(
            dbname="postgresql_instance_free",
            user="postgresql_instance_free_user",
            password="bz3SdnKi6g6TRdM4j1AtE2Ash8VNgiQO",
            host="dpg-cts4psa3esus73dn1cn0-a.oregon-postgres.render.com",
            port="5432"
        )
        cursor = conn.cursor()

        # Insert the new word into the prohibited_words table
        sql = """INSERT INTO prohibited_words (word, alternative) VALUES (%s, %s);"""
        cursor.execute(sql, (word.lower(), alternative or "No alternative provided"))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:

        # If the DB insertion fails, you might still consider the JSON save successful
        return jsonify({
            'success': False,
            'error': f'Could not save to DB: {str(e)}. JSON save succeeded.'
        }), 500


    # 3) Return success
    return jsonify({
        'success': True,
        'message': f'Prohibited word "{word}" added to JSON and DB successfully.'
    }), 200

# Flagging Prohibited Words
@app.route('/flag_prohibited_words', methods=['POST'])
def flag_prohibited_words():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"success": False, "error": "Text to scan is required."}), 400
        flagged_instances = []

        # Scan text for prohibited words
        for entry in prohibited_words:
            word = entry["word"]
            if word.lower() in text.lower():
                flagged_instances.append({
                    "word": word,
                    "alternative": entry.get("alternative", "No alternative provided"),
                    "context": get_context(text, word)
                })
        return jsonify({"success": True, "flagged": flagged_instances})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def get_context(text, word, window=30):
    """Return a snippet of text around the flagged word for context."""
    idx = text.lower().find(word.lower())
    if idx == -1:
        return None
    start = max(idx - window, 0)
    end = min(idx + len(word) + window, len(text))
    return text[start:end]








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
        
        return jsonify({"success": True, "message": "Advisor type saved successfully"})
    except Exception as e:
        logger.error(f"Error saving advisor type: {e}")
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



# Control Panel
@app.route('/control-panel')
def control_panel():
    return redirect(url_for('manage_disclosures'))

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


# Checking Subscription
@app.route('/api/check_subscription', methods=['GET'])
def check_subscription():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email parameter is required"}), 400

    # Sync subscription status with Stripe
    user = sync_subscription_status(email)
    if "error" in user:
        return jsonify(user), 404

    if user.get('Subscribed') == "No":
        return jsonify({"showPaymentPopup": True})
    return jsonify({"showPaymentPopup": False})



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

# Function to cancel subscription in PROFILE HTMl@app.route('/api/cancel-subscription', methods=['POST'])
@app.route('/api/cancel-subscription', methods=['POST'])
def cancel_subscription():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email is required.'}), 400

    try:
        # Retrieve the Stripe customer associated with the email
        customers = stripe.Customer.list(email=email).data
        if not customers:
            return jsonify({'error': 'Customer not found.'}), 404
        customer = customers[0]

        # Retrieve active subscriptions for the customer
        subscriptions = stripe.Subscription.list(customer=customer.id, status="active").data
        if not subscriptions:
            return jsonify({'error': 'No active subscription found for user.'}), 404

        # Cancel the active subscription
        subscription = subscriptions[0]
        stripe.Subscription.modify(
            subscription.id,              # <-- pass the actual subscription ID
            cancel_at_period_end=True
        )


        # Sync subscription status locally
        user = sync_subscription_status(email)
        if "error" in user:
            return jsonify(user), 404

        return jsonify({'message': 'Subscription canceled successfully.'})
    except Exception as e:
        logger.info(f"Error canceling subscription: {str(e)}")
        return jsonify({'error': 'An error occurred while canceling the subscription.'}), 500


# Function to track users who have unsubscribed
@app.route('/update-unsubscribed', methods=['POST'])
def update_unsubscribed():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # Sync subscription status locally
    user = sync_subscription_status(email)
    if "error" in user:
        return jsonify(user), 404

    return jsonify({'message': 'User unsubscribed successfully.'})

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







# STRIPE FUNCTIONALITY


# Live Stripe Key
#stripe.api_key = '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'


# Test Stripe Key
stripe.api_key = 'sk_test_51GbFkxE4vPtJSn6ecxnMRVMsiWwgJrxOWxsQ7rSuA1s5NtTYEyRKyNr7Bi575PlHwTqng7gcI92coYA5UpBwu0xT00f24K6saO'


@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/create-subscription', methods=['POST'])
def create_subscription():
    try:
        data = request.get_json()
        payment_method_id = data.get('paymentMethodId')
        price_id = data.get('priceId')
        email = data.get('email')  # Get the email sent from the frontend

        # Get billing address details
        billing_details = {
            'name': data.get('billingName'),
            'email': email,  # Use the email from the frontend
            'address': {
                'line1': data.get('billingAddress'),
                'city': data.get('billingCity'),
                'state': data.get('billingState'),
                'postal_code': data.get('billingZip'),
                'country': data.get('billingCountry'),
            },
        }
        if not email:
            return jsonify({'error': 'Email is required.'}), 400

        # Create the customer
        customer = stripe.Customer.create(
            email=email,
        )

        # Attach the PaymentMethod to the customer
        stripe.PaymentMethod.attach(
            payment_method_id,
            customer=customer.id,
        )

        # Update the billing details on the PaymentMethod
        stripe.PaymentMethod.modify(
            payment_method_id,
            billing_details=billing_details,
        )

        # Set the default payment method for the customer
        stripe.Customer.modify(
            customer.id,
            invoice_settings={'default_payment_method': payment_method_id},
        )
        
        # Create the subscription
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{'price': price_id}],
            expand=['latest_invoice.payment_intent'],
        )

        # Get the PaymentIntent if it exists
        payment_intent = subscription.latest_invoice.payment_intent
        client_secret = payment_intent.client_secret if payment_intent else None

        

        # Update the user's subscription status in users.json
        users = load_users()
        user = next((u for u in users if u['email'] == email), None)
        if user:
            user['Subscribed'] = "Yes"
            user['Unsubscribed'] = "No"  # Mark as subscribed

            if payment_intent and payment_intent.status == 'succeeded':
                user["Sign Up Status"] = "Signed Up"
            
            save_users(users)  # Save the updated users back to the file

        return jsonify({
            'subscriptionId': subscription.id,
            'clientSecret': client_secret,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/create-payment-intent', methods=['POST'])
def create_payment_intent():
    try:
        # Get payment amount from the client-side
        data = request.get_json()
        amount = data['amount']  # Amount in cents

        # Create a payment intent
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='usd',
            automatic_payment_methods={'enabled': True},
        )
        return jsonify({'clientSecret': intent['client_secret']})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/charge-card', methods=['POST'])
def charge_card():
    try:
        data = request.get_json()
        price_id = data.get('priceId')
        email = data.get('email')  # Email passed from the frontend
        if not price_id or not email:
            return jsonify({'error': 'Invalid price ID or email'}), 400

        # Map price IDs to their respective amounts in cents
        price_mapping = {
            'price_1Qf8LCE4vPtJSn6exafXQTA8': 2500,   # $25 in cents
            'price_1Qf8O4E4vPtJSn6enzkQ08eC': 5000,   # $50 in cents
            'price_1Qf8QCE4vPtJSn6e6eTX5kTE': 10000,  # $100 in cents
        }
        amount = price_mapping.get(price_id)
        if not amount:
            return jsonify({'error': 'Invalid price ID'}), 400

        # Retrieve the customer from Stripe using email
        customers = stripe.Customer.list(email=email).data
        if not customers:
            return jsonify({'error': 'No customer found with this email'}), 404
        customer = customers[0]  # Assume the first customer is correct

        # Retrieve the customer's default payment method
        payment_methods = stripe.PaymentMethod.list(
            customer=customer.id,
            type="card",
        )

        if not payment_methods.data:
            return jsonify({'error': 'No saved payment method found for this customer'}), 404

        # Use the first payment method
        default_payment_method = payment_methods.data[0]
        
        # Create a payment intent
        payment_intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='usd',
            customer=customer.id,
            payment_method=default_payment_method.id,
            confirm=True,
            description=f'Purchase of usage credits ({price_id})',
            automatic_payment_methods={
                "enabled": True,
                "allow_redirects": "never"
            },
        )

        # If payment succeeded, update the user's usage denominator
        users = load_users()
        user = next((u for u in users if u['email'] == email), None)
        if user:

            # Initialize usage_denominator if not present
            if 'usage_denominator' not in user:
                user['usage_denominator'] = 5.0

            # 1. Add the number purchased to usage_denominator
            if price_id == 'price_1Qf8LCE4vPtJSn6exafXQTA8':    # $25
                user['usage_denominator'] += 25
            elif price_id == 'price_1Qf8O4E4vPtJSn6enzkQ08eC':  # $50
                user['usage_denominator'] += 50
            elif price_id == 'price_1Qf8QCE4vPtJSn6e6eTX5kTE':  # $100
                user['usage_denominator'] += 100

            # 2. Subtract the current usage from the new denominator
            user['usage_denominator'] -= user['usage']

            # 3. Ensure usage remains unchanged
            user['usage'] = user.get('usage', 0.0)
            save_users(users)

        return jsonify({'clientSecret': payment_intent.client_secret})
    except stripe.error.CardError as e:
        return jsonify({'error': f"Card error: {e.user_message}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Fetches the next billing date the user can expect from STRIPE
@app.route('/api/next-billing-date', methods=['GET'])
def get_next_billing_date():
    email = request.args.get('email')
    if not email:
        return jsonify({'error': 'Email parameter is required'}), 400

    # 1. Fetch the Stripe customer by email
    customer = stripe.Customer.list(email=email).data
    if not customer:
        return jsonify({'error': 'Customer not found'}), 404
    customer_id = customer[0].id

    # 2. Check for an ACTIVE subscription first
    active_subscriptions = stripe.Subscription.list(
        customer=customer_id,
        status='active'
    ).data

    if active_subscriptions:
        # The user still has an active subscription (including "cancel_at_period_end" but not ended yet)
        subscription = active_subscriptions[0]
        next_billing_date = subscription.current_period_end

        # Convert timestamp to YYYY-MM-DD
        formatted_date = datetime.utcfromtimestamp(next_billing_date).strftime('%Y-%m-%d')
        logger.info(f"Next Billing Date (raw): {next_billing_date}")

        # Update AccessEndDate in users.json
        users = load_users()
        user = next((u for u in users if u['email'] == email), None)
        if user:
            user['AccessEndDate'] = formatted_date
            save_users(users)
            logger.info(f"Updated AccessEndDate for {email}: {formatted_date}")

        # Return the next billing date as you normally do
        return jsonify({'nextBillingDate': next_billing_date})

    # ----------------------------------------------------------------------
    # 3. If there's NO active subscription, check for a CANCELLED subscription
    canceled_subscriptions = stripe.Subscription.list(
        customer=customer_id,
        status='canceled',
        limit=1  # grab the most recent canceled subscription
    ).data

    if canceled_subscriptions:
        subscription = canceled_subscriptions[0]

        # If canceled_at_period_end was used, once it's truly ended,
        # ended_at will be set to the date/time it ended.
        ended_at = subscription.ended_at
        cancel_at_period_end = subscription.cancel_at_period_end
        current_period_end = subscription.current_period_end

        # Decide which date to show:
        # - If ended_at exists, use it (the subscription is truly ended).
        # - If ended_at is None but cancel_at_period_end is True, it normally means
        #   the subscription is still active until current_period_end. But if Stripe
        #   lists it here under "canceled," it's presumably ended, so ended_at should be set.
        if ended_at:
            # ended_at is a Unix timestamp for final cancellation
            date_to_use = ended_at
        else:
            # Fallback to current_period_end if ended_at is somehow missing
            date_to_use = current_period_end

        canceled_date_str = datetime.utcfromtimestamp(date_to_use).strftime('%Y-%m-%d')

        # Update AccessEndDate to reflect the final day they actually had/have access
        users = load_users()
        user = next((u for u in users if u['email'] == email), None)
        if user:
            user['AccessEndDate'] = canceled_date_str
            save_users(users)
            logger.info(f"Updated AccessEndDate for {email} to canceled date: {canceled_date_str}")

        return jsonify({
            'message': 'User subscription is canceled',
            'endedAt': date_to_use
        })

    # 4. If we have no active sub and no canceled sub in Stripe, return an error
    return jsonify({'error': 'No subscription found for this user.'}), 404
    
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
    if 'user_email' not in session:
        # User is not logged in, redirect to intro page
        return redirect(url_for('intro_page'))
    
    # Log the currently logged-in user
    logged_in_user = session.get('user_email')
    logger.info(f"Logged-in user: {session.get('user_email')}")
    
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

    # Render the upload page and pass the logged-in user to the template
    return render_template('upload.html', user_email=logged_in_user)


# DOES NOT WORK YET
# Function to delete archived text in JSON file for archives

from datetime import datetime

@app.route('/delete_archived_text', methods=['POST'])
def delete_archived_text():
    try:
        data = request.get_json()
        date = data.get('date')  # Use the `date` field to identify the row
        logger.info(f"Date received for deletion: {date}")  # Debugging
        if not date:
            return jsonify({'error': 'Missing date field'}), 400

        # Load existing archives
        if os.path.exists(ARCHIVES_FILE):
            with open(ARCHIVES_FILE, 'r') as f:
                archives = json.load(f)
            logger.info(f"Archives before deletion: {archives}")  # Debugging

            # Normalize date formats for comparison
            try:
                received_date = datetime.strptime(date, "%m/%d/%Y").strftime('%Y-%m-%d')
            except ValueError as e:
                logger.info(f"Error parsing date: {e}")
                return jsonify({'error': 'Invalid date format received'}), 400

            # Filter out the entry to delete
            updated_archives = [
                entry for entry in archives
                if not entry.get('date', '').startswith(received_date)  # Check by matching the start of the date
            ]
            logger.info(f"Archives after deletion: {updated_archives}")  # Debugging

            # Save the updated archives
            with open(ARCHIVES_FILE, 'w') as f:
                json.dump(updated_archives, f, indent=4)
            return jsonify({'success': True, 'message': 'Row deleted successfully!'})
        else:
            return jsonify({'error': 'Archives file not found'}), 404
    except Exception as e:
        logger.info(f"Error in delete_archived_text: {e}")
        return jsonify({'error': str(e)}), 500





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




#Proces File Route
@app.route('/process/<filename>', methods=['GET'])
def process_file(filename):
    global processed_files
    logger.info(f"Processing file: {filename}")
    logger.info(f"File size: {os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename))} bytes")

    logger.info(f"[{datetime.utcnow()}] Incoming request: /process/{filename}")

    # Check if user is admin (default to False if not logged in)
    is_admin = False
    if 'user_email' in session:
        user_email = session['user_email']
        users = load_users()
        user = next((u for u in users if u['email'] == user_email), None)
        if user:
            is_admin = user.get('Administrator Access NEW') == 'Yes'

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
            results=processed_files[filename].get('results', 0),
            is_admin=is_admin,  # Pass is_admin to the template
            filename=filename
        )

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        logger.info(f"File {filename} not found.")
        final_check_status[filename] = False
        return render_template('results.html', text="Error: File not found", revised_text="", disclosures=[], sliced_disclosures=[], finra_analysis=[], results=0, is_admin=is_admin)

    try:
        # Extract text by page based on file type
        if file_path.endswith('.pdf'):
            text_by_page = extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text_by_page = extract_text_from_docx(file_path)
        elif file_path.endswith('.mp4'):
            # Handle video files
            logger.info(f"Transcribing MP4 file: {file_path}")
            text_by_page = transcribe_mp4(file_path)
        else:
            logger.info(f"Unsupported file type: {file_path}")
            final_check_status[filename] = False
            return render_template('results.html', text="Unsupported file type", revised_text="", disclosures=[], sliced_disclosures=[], finra_analysis=[], results=0, is_admin=is_admin)

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

        # Add this at the end, before returning the template
        log_session_token_summary()

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
            filename=filename,  # Pass filename to template
            is_admin=is_admin  # Pass is_admin to the template
        )

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        final_check_status[filename] = False
        return render_template('results.html', text="Error processing file", revised_text="", disclosures=[], sliced_disclosures=[], finra_analysis=[], results=0, is_admin=is_admin)
    
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

# AI-based function to detect references to economic forecasts
#def ai_detects_economic_forecast(text):
    # Prompt the AI to detect economic forecasts
    #prompt = f"Does the following content reference an economic forecast?\n\nContent: {text}"
    
    #try:
        #response = anthropic.messages.create(
            #model="claude-3-opus-20240229",
            #system=SYSTEM_MESSAGE,  # Using global system message
            #max_tokens=50,
            #messages=[
                #{
                    #"role": "user",
                    #"content": prompt
                #}
            #]
        #)
        #ai_response = response.content[0].text.strip().lower()
        #return 'yes' in ai_response or 'economic forecast' in ai_response or 'prediction' in ai_response
    #except Exception as e:
        #logger.error(f"Error detecting economic forecast: {e}")
        #return False



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
import json
import random
import re
from difflib import SequenceMatcher

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

# Load prohibited words from JSON
def load_prohibited_words_custom():
    try:
        with open('prohibited_words.json', 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.info(f"Error loading prohibited words: {e}")
        return []





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
        BERT_MODEL, BERT_TOKENIZER = get_bert_model()
        
        # First, run the text through BERT
        inputs = BERT_TOKENIZER(custom_text, 
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
        
        logger.info(f"BERT prediction for custom alternative: {prediction} ('{'non-compliant' if prediction == 1 else 'compliant'}'), Confidence: {confidence:.1%}")
        
        # If BERT predicts compliant (class 0), return success immediately
        if prediction == 0:
            return jsonify({
                "compliant": True,
                "message": "No issues identified.",
                "flagged_instances": []
            }), 200
            
        # If BERT predicts non-compliant (class 1), verify with Deepseek API
        verification_prompt = f"""
Determine if this text violates FINRA's communication rules by being false, misleading, promissory or exaggerated:

"{custom_text}"

Answer with ONLY "YES" or "NO", followed by a brief explanation.
"YES" means it IS non-compliant.
"NO" means it IS compliant.

IMPORTANT DISTINCTION:
- Non-compliant: Statements that present specific financial benefits, tax advantages, or performance outcomes as definite facts without proper qualifiers
- Compliant: General statements, subjective opinions, or descriptions that don't make specific claims about financial outcomes or benefits

CRITICAL: Statements presented as definitive facts without qualifying language are typically non-compliant when they involve:
- Tax benefits
- Investment outcomes
- Financial advantages
- Product features that don't universally apply

Examples of non-compliant statements:
- "A Traditional IRA is a place to put your money to save on taxes." (presents tax saving as definite)
- "Roth IRAs are a great vehicle for tax free investing" (presents absolute statement when additioal rules apply to receive benefits of investing in ROTH IRA)
- "IRAs are vehicles with tax advantages" (not necessarily true in all cases)
- "This fund outperforms the market" (absolute claim without qualification)
- "The strategy protects your assets during downturns" (unqualified protection claim)

Examples of compliant alternatives:
- "A Traditional IRA is a place to put your money to potentially save on taxes."
- "Roth IRAs may offer tax advantages for qualifying investors" or "Roth IRAs are a potential vehicle for tax advantaged investing"
- "IRAs are vehicles with potential tax advantages" (clarifies all advantages don't apply to everyone)
- "This fund is designed to seek competitive returns relative to its benchmark"
- "The strategy aims to help manage risk during market downturns"

Always answer "YES" (non-compliant) for statements that:
1. Present possible, implied or conditional benefits as definite outcomes
2. Make absolute claims about tax advantages
3. Lack qualifiers like "may," "potential," "designed to," or "aims to" when discussing benefits
4. State as fact something that doesn't apply in all circumstances

All financial benefits and advantages MUST be qualified with appropriate language.
"""

        
        verification_response = call_deepseek_api(verification_prompt)
        logger.info(f"Deepseek verification response: {verification_response}")
        
        # Check if Deepseek confirms this is non-compliant
        is_non_compliant = False
        if verification_response:
            first_word = verification_response.strip().split()[0].upper() if verification_response.strip() else ""
            is_non_compliant = first_word == "YES"
        
        if is_non_compliant:
            # Send another prompt to categorize the violation types
            categorization_prompt = f"""
            Identify the type of FINRA rule violation in this text:
            
            "{custom_text}"
            
            Choose ONLY ONE primary violation type from this list:
            1. Promissory/Guarantee (making promises or guarantees about performance)
            2. Exaggeration (overstating benefits or capabilities)
            3. Misleading (presenting information in a way that could mislead investors)
            4. Unbalanced (not giving equal weight to risks and benefits)
            
            Respond with ONLY the number and name of the primary violation type.
            """
            
            violation_type_response = call_deepseek_api(categorization_prompt).strip()
            logger.info(f"Violation type categorization: {violation_type_response}")
            
            # Get a compliant alternative using Deepseek
            alternative_prompt = f"""
            This text violates FINRA rules: "{custom_text}"
            
            Rewrite it to be compliant by:
            1. Removing absolute statements
            2. Adding appropriate qualifiers
            3. Avoiding guarantees or promises
            4. Maintaining the original meaning
            
            Respond with ONLY the compliant alternative text and nothing else.
            """
            
            compliant_text = call_deepseek_api(alternative_prompt).strip()
            
            # Clean quotes if present
            if compliant_text.startswith('"') and compliant_text.endswith('"'):
                compliant_text = compliant_text[1:-1]
            
            # Provide a more specific rationale based on the violation type
            if "1" in violation_type_response or "Promissory" in violation_type_response or "Guarantee" in violation_type_response:
                customized_rationale = "This text may not comply with FINRA regulations as it appears to make promises or guarantees about performance. Consider using qualifying language like 'may' instead of 'will' and avoiding absolute statements."
            
            elif "2" in violation_type_response or "Exaggeration" in violation_type_response:
                customized_rationale = "This text may not comply with FINRA regulations as it appears to contain exaggerated claims. Consider toning down superlatives and providing a more balanced view of potential outcomes."
                        
            elif "3" in violation_type_response or "Misleading" in violation_type_response:
                customized_rationale = "This text may not comply with FINRA regulations as it could potentially mislead investors. Consider providing more context and ensuring all claims are accurately represented."
            
            elif "4" in violation_type_response or "Unbalanced" in violation_type_response:
                customized_rationale = "This text may not comply with FINRA regulations as it appears to present an unbalanced view. Consider giving equal weight to both risks and benefits to provide a more complete picture."
            
            else:
                # Fallback if categorization is unclear
                customized_rationale = "This text may not comply with FINRA regulations. Consider adding appropriate qualifiers, removing absolute statements, and ensuring balanced presentation of risks and benefits."
            
            # Create a flagged instance in the expected format
            flagged_instance = {
                "flagged_instance": custom_text,
                "compliance_status": "non-compliant",
                "specific_compliant_alternative": compliant_text,
                "rationale": customized_rationale,  # Use customized rationale here
                "confidence": f"{confidence:.1%}"
            }
            
            return jsonify({
                "compliant": False,
                "message": "Compliance issues found.",
                "flagged_instances": [flagged_instance]
            }), 200
        else:
            # Deepseek found it compliant, override BERT's prediction
            return jsonify({
                "compliant": True,
                "message": "No issues identified.",
                "flagged_instances": []
            }), 200
            
    except Exception as e:
        logger.error(f"Error during custom text compliance check: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "compliant": False,
            "error": "An error occurred during compliance checking.",
            "message": str(e),
            "flagged_instances": []
        }), 500
    
    
# Assessing Compliance Level
@app.route('/assess_compliance_level', methods=['POST'])
def assess_compliance_level():
    data = request.get_json()
    custom_text = data.get("custom_text")  # Get the custom text instead of feedback

    # Clear, concise prompt for AI to determine compliance level
    prompt = f"""

    Evaluate the following text for compliance with FINRA and SEC marketing rules:

    - Avoid implications of guarantees, guarantees of positive returns, or implications of risk-free investing.

    - Statements can be compliant even if they do not explicitly acknowledge risks, as long as they do not promise gains or make misleading claims.

    - If the text is compliant, respond with "compliant".

    - If the text is non-compliant, respond with "non-compliant".


    Text: "{custom_text}"

    Answer only with "compliant" or "non-compliant". Provide a single-sentence rationale.

    """
    compliance_level = call_deepseek_api(prompt).strip().lower()
    # Ensure we only get compliant/non-compliant
    if compliance_level.startswith("compliant") or compliance_level.startswith("non-compliant"):
        compliance_status = compliance_level.split()[0]  # Get just the first word
    else:
        compliance_status = "error"  # Or handle unexpected responses differently

    return jsonify({"compliance_level": compliance_status})




# WTF is this???

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

# ARCHIVES PAGE FUNCTIONALITY

ARCHIVES_FILE = "archives.json"

# Route to save archived text
@app.route('/save_archived_text', methods=['POST'])
def save_archived_text():
    try:
        data = request.get_json()
        
        # Ensure required fields are present
        required_fields = ['text', 'user_email', 'link', 'disclosures', 'distribution_channel', 'distribution_method']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Add a timestamp for the date field
        data['date'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # Load existing archives
        archives = []
        if os.path.exists(ARCHIVES_FILE):
            with open(ARCHIVES_FILE, 'r') as f:
                archives = json.load(f)

        # Add the new entry
        archives.append(data)

        # Save the updated archives
        with open(ARCHIVES_FILE, 'w') as f:
            json.dump(archives, f, indent=4)
        return jsonify({'success': True})
    except Exception as e:
        logger.info(f"Error in save_archived_text: {e}")
        return jsonify({'error': str(e)}), 500

    # Append the new entry with disclosures
    distribution_channels = data.get('distribution_channel', [])  # Correct key matching UPLOADS HTML
    logger.info("Received Distribution Channels:", distribution_channels)  # Debugging
    distribution_methods = data.get('distribution_method', [])  # Get selected distribution methods
    logger.info("Distribution Channels:", distribution_channels)  # Confirm this value
    logger.info("Distribution Methods:", distribution_methods)
    archives.append({

        "date": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        "text": text,
        "link": file_link,
        "disclosures": disclosures,
        "distribution_channels": distribution_channels,  # Save distribution channels
        "distribution_methods": distribution_methods     # Save distribution methods
    })

    with open(ARCHIVES_FILE, 'w') as f:
        json.dump(archives, f, indent=4)
    return jsonify({"message": "Text archived successfully!"})


# Route to get archived texts

@app.route('/get_archived_texts', methods=['GET'])
def get_archived_texts():
    if os.path.exists(ARCHIVES_FILE):
        with open(ARCHIVES_FILE, 'r') as f:
            archives = json.load(f)
        logger.info("Archived Data (All):", archives)  # Debug
        if 'user_email' in session:
            user_email = session['user_email']
            logger.info(f"Logged-in User Email: {user_email}")  # Debug

            # Filter archives for the logged-in user
            filtered_archives = [entry for entry in archives if entry.get('user_email') == user_email]
            logger.info(f"Filtered Archives for {user_email}:", filtered_archives)  # Debug
            return jsonify(filtered_archives)
        else:

            # No user logged in: Return all for testing/admin purposes
            logger.info("No user logged in. Returning all archives.")  # Debug
            return jsonify(archives)
    else:
        logger.info("Archives file does not exist.")  # Debug
        return jsonify([])



@app.route('/archived', methods=['GET'])
def archived():
    return render_template('archived.html')










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

    # Special handling for MP4 files
    if filename.lower().endswith('.mp4'):
        # Trigger the video processing endpoint
        return jsonify({
            'filename': filename,
            'is_mp4': True,
            'message': 'MP4 file uploaded, processing will begin'
        }), 200
    
    # Return the filename for progress tracking (for non-MP4 files)
    return jsonify({'filename': filename}), 200


# Second endpoint to get rationales based on check_id
@app.route('/get_rationales/<check_id>', methods=['POST'])
def get_rationales(check_id):
    data = request.get_json()
    custom_text = data.get("text", "").strip()
    
    # Rest of your existing check_quick_text logic to generate rationales
    # ...


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
        BERT_MODEL, BERT_TOKENIZER = get_bert_model()
        
        # Split text into sentences for BERT analysis
        sentences = split_into_sentences(custom_text)
        flagged_instances = []
        bert_flagged_sentences = []
        
        # First pass: Use BERT to identify potentially non-compliant sentences
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
            
            logger.info(f"BERT prediction for sentence: {prediction} ('{'non-compliant' if prediction == 1 else 'compliant'}'), Confidence: {confidence:.2f}")
            
            # If BERT predicts non-compliant (class 1)
            if prediction == 1 and len(sentence.strip()) > 5:
                bert_flagged_sentences.append({
                    "sentence": sentence,
                    "confidence": confidence
                })
        
        # Second pass: For each BERT-flagged sentence, verify with Deepseek API
        for flagged in bert_flagged_sentences:
            sentence = flagged["sentence"]
            confidence = flagged["confidence"]
            
            # Verify with Deepseek if sentence is truly non-compliant
            verification_prompt = f"""
Determine if this text violates FINRA's communication rules by being false, misleading, promissory or exaggerated:

"{sentence}"

Answer with ONLY "YES" or "NO", followed by a brief explanation.
"YES" means it IS non-compliant.
"NO" means it IS compliant.

IMPORTANT DISTINCTION:
- Non-compliant: Statements that present specific financial benefits, tax advantages, or performance outcomes as definite facts without proper qualifiers
- Compliant: General statements, subjective opinions, or descriptions that don't make specific claims about financial outcomes or benefits

CRITICAL: Statements presented as definitive facts without qualifying language are typically non-compliant when they involve:
- Tax benefits
- Investment outcomes
- Financial advantages
- Product features that don't universally apply

Examples of non-compliant statements:
- "A Traditional IRA is a place to put your money to save on taxes." (presents tax saving as definite)
- "Roth IRAs are a great vehicle for tax free investing" (presents absolute statement when additioal rules apply to receive benefits of investing in ROTH IRA)
- "IRAs are vehicles with tax advantages" (not necessarily true in all cases)
- "This fund outperforms the market" (absolute claim without qualification)
- "The strategy protects your assets during downturns" (unqualified protection claim)

Examples of compliant alternatives:
- "A Traditional IRA is a place to put your money to potentially save on taxes."
- "Roth IRAs may offer tax advantages for qualifying investors" or "Roth IRAs are a potential vehicle for tax advantaged investing"
- "IRAs are vehicles with potential tax advantages" (clarifies all advantages don't apply to everyone)
- "This fund is designed to seek competitive returns relative to its benchmark"
- "The strategy aims to help manage risk during market downturns"

Always answer "YES" (non-compliant) for statements that:
1. Present possible, implied or conditional benefits as definite outcomes
2. Make absolute claims about tax advantages
3. Lack qualifiers like "may," "potential," "designed to," or "aims to" when discussing benefits
4. State as fact something that doesn't apply in all circumstances

All financial benefits and advantages MUST be qualified with appropriate language.
"""
            #verification_prompt = f"""
            #Determine if this text is overstated or violates FINRA's communication rules around being false, misleading, promissory or exaggerated:
            
            #"{sentence}"
            
            #Answer with ONLY "YES" or "NO", followed by a brief explanation.
            #"YES" means it IS non-compliant.
            #"NO" means it IS compliant.
            #"""
            
            verification_response = call_deepseek_api(verification_prompt)
            logger.info(f"Deepseek verification response: {verification_response}")
            
            # Check if Deepseek confirms this is non-compliant
            is_non_compliant = False
            if verification_response:
                first_word = verification_response.strip().split()[0].upper() if verification_response.strip() else ""
                is_non_compliant = first_word == "YES"
            
            if is_non_compliant:
                # If confirmed non-compliant, get a category and rationale
                categorization_prompt = f"""
                Identify the type of FINRA rule violation in this text:
                
                "{sentence}"
                
                Choose ONLY ONE primary violation type from this list:
                1. Promissory/Guarantee (making promises or guarantees about performance)
                2. Exaggeration (overstating benefits or capabilities)
                3. Misleading (presenting information in a way that could mislead investors)
                4. Unbalanced (not giving equal weight to risks and benefits)
                
                Respond with ONLY the number and name of the primary violation type.
                """
                
                violation_type_response = call_deepseek_api(categorization_prompt).strip()
                logger.info(f"Violation type categorization: {violation_type_response}")
                
                # Get a compliant alternative using Deepseek
                alternative_prompt = f"""
                This text violates FINRA rules: "{sentence}"
                
                Rewrite it to be compliant by:
                1. Removing absolute statements
                2. Adding appropriate qualifiers
                3. Avoiding guarantees or promises
                4. Maintaining the original meaning
                
                Respond with ONLY the compliant alternative text and nothing else.
                """
                
                compliant_text = call_deepseek_api(alternative_prompt).strip()
                
                # Clean quotes if present
                if compliant_text.startswith('"') and compliant_text.endswith('"'):
                    compliant_text = compliant_text[1:-1]
                
                # Generate a specific rationale based on the violation type
                if "1" in violation_type_response or "Promissory" in violation_type_response or "Guarantee" in violation_type_response:
                    customized_rationale = "This text contains promises or guarantees about performance which violates FINRA regulations. Consider using qualifying language like 'may' instead of 'will' and avoiding absolute statements."
                    category = "Promissory/Guarantee"
                elif "2" in violation_type_response or "Exaggeration" in violation_type_response:
                    customized_rationale = "This text contains exaggerated claims which violates FINRA regulations. Consider toning down superlatives and providing a more balanced view of potential outcomes."
                    category = "Exaggeration"
                elif "3" in violation_type_response or "Misleading" in violation_type_response:
                    customized_rationale = "This text could potentially mislead investors in a way that violates FINRA regulations. Consider providing more context and ensuring all claims are accurately represented."
                    category = "Misleading"
                elif "4" in violation_type_response or "Unbalanced" in violation_type_response:
                    customized_rationale = "This text presents an unbalanced view of risks and benefits which violates FINRA regulations. Consider giving equal weight to both risks and benefits to provide a more complete picture."
                    category = "Unbalanced"
                else:
                    customized_rationale = "This text may not comply with FINRA regulations. Consider adding appropriate qualifiers, removing absolute statements, and ensuring balanced presentation of risks and benefits."
                    category = "General Compliance Issue"
                
                # Add to flagged instances
                instance = {
                    "flagged_instance": sentence,
                    "compliance_status": "non-compliant",
                    "specific_compliant_alternative": compliant_text,
                    "rationale": customized_rationale,
                    "confidence": f"{confidence:.1%}",
                    "category": category
                }
                flagged_instances.append(instance)
        
        # Update user usage if needed
        if 'user_email' in session:
            # Calculate a cost based on number of API calls to Deepseek
            api_calls = len(bert_flagged_sentences)
            cost_per_call = 0.01  # $0.01 per API call - adjust as needed
            update_user_usage(session['user_email'], usage_increment=api_calls * cost_per_call)
            logger.info(f"Updated usage for {session['user_email']}: ${api_calls * cost_per_call:.2f} (for {api_calls} API calls)")
        
        # Determine overall compliance based on flagged instances
        is_compliant = len(flagged_instances) == 0
        
        # Create a specific message based on the flagged instances
        message = ""
        if is_compliant:
            message = "No compliance issues found. This text appears to be compliant with FINRA regulations."
        else:
            # Get the categories of issues found
            categories = set(instance.get("category", "General Compliance Issue") for instance in flagged_instances)
            
            # Create a message based on the categories
            if len(categories) == 1:
                category = next(iter(categories))
                message = f"Found potential {category} compliance issue. Please review the suggested alternatives."
            else:
                category_list = ", ".join(list(categories)[:-1]) + " and " + list(categories)[-1] if len(categories) > 1 else next(iter(categories))
                message = f"Found potential compliance issues related to {category_list}. Please review the suggested alternatives."
        
        # Log token usage summary
        log_session_token_summary()
        
        return jsonify({
            "compliant": is_compliant,
            "message": message,
            "flagged_instances": flagged_instances
        }), 200
        
    except Exception as e:
        logger.error(f"Error during quick text compliance check: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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

            # For MP4 files, handle transcription with progress updates
            if filename.endswith('.mp4'):
                # Start transcription in a separate thread to allow progress updates during processing
                import threading
                transcription_complete = threading.Event()
                transcription_result = [None]  # Use a list to store the result
                
                def do_transcription():
                    try:
                        result = transcribe_mp4(file_path)
                        transcription_result[0] = result
                        transcription_complete.set()
                    except Exception as e:
                        logger.error(f"Error in transcription thread: {e}")
                        transcription_complete.set()  # Signal completion even on error
                
                # Start transcription in background
                threading.Thread(target=do_transcription).start()
                
                # Show incremental progress during transcription (1% every 3 seconds up to 20%)
                current_progress = 0
                last_keepalive = time.time()
                
                while current_progress < 20 and not transcription_complete.is_set():
                    # Increment progress
                    current_progress += 1
                    yield f"data: {current_progress:.2f}\n\n"
                    
                    # Send keepalive messages every 15 seconds to prevent connection timeouts
                    for _ in range(3):  # Split wait time into smaller chunks
                        if transcription_complete.is_set():
                            break
                        
                        # Wait for a shorter period (1 second)
                        transcription_complete.wait(1)
                        
                        # Send keepalive message if needed
                        now = time.time()
                        if now - last_keepalive > 15:
                            yield f"data: {current_progress:.2f}\n\n"  # Send same progress as keepalive
                            last_keepalive = now
                
                # If we exit the loop before reaching 20%, jump to 20%
                if current_progress < 20:
                    current_progress = 20
                    yield f"data: {current_progress:.2f}\n\n"
                
                # Wait for transcription to complete if it hasn't already
                timeout_counter = 0
                max_timeout = 300  # 5 minutes max wait
                
                while not transcription_complete.is_set() and timeout_counter < max_timeout:
                    # Wait a bit but keep connection alive
                    time.sleep(2)
                    timeout_counter += 2
                    
                    # Send keepalive periodically
                    now = time.time()
                    if now - last_keepalive > 15:
                        yield f"data: {current_progress:.2f}\n\n"  # Send same progress as keepalive
                        last_keepalive = now
                
                # If we timed out waiting for transcription
                if timeout_counter >= max_timeout and not transcription_complete.is_set():
                    yield f"data: Error: Transcription is taking too long, please check server logs\n\n"
                    return
                
                # Get the transcription result
                text_by_page = transcription_result[0]
                if text_by_page is None:
                    yield f"data: Error: Transcription failed\n\n"
                    return
            else:
                # For other file types, proceed as normal
                if filename.endswith('.pdf'):
                    text_by_page = extract_text_from_pdf(file_path)
                elif filename.endswith('.docx'):
                    text_by_page = extract_text_from_docx(file_path)
                else:
                    yield f"data: Error: Unsupported file type\n\n"
                    return
                
                # Start with 0% for non-MP4 files
                current_progress = 0.0
                
            # The rest of the function continues as before
            all_text = " ".join([page["text"] for page in text_by_page])
            sentences = split_into_sentences(all_text)
            flagged_instances = []  # Define flagged instances if necessary
            total_sentences = len(sentences)

            if total_sentences == 0:
                yield f"data: Error: No sentences to process\n\n"
                return

            # Simulate progress from current_progress to 60%
            remaining_progress = 60 - current_progress
            increment = remaining_progress / total_sentences if total_sentences > 0 else 0
            last_keepalive = time.time()
            
            for i, sentence in enumerate(sentences, start=1):
                # Use a smaller sleep time for faster response
                time.sleep(0.5)  # Simulate processing time
                
                progress = current_progress + (i * increment)
                yield f"data: {progress:.2f}\n\n"
                
                # Send keepalive if needed
                now = time.time()
                if now - last_keepalive > 15:
                    yield f"data: {progress:.2f}\n\n"  # Resend as keepalive
                    last_keepalive = now

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
            last_keepalive = time.time()
            timeout_counter = 0
            max_wait_time = 300  # 5 minutes max wait
            
            while not final_check_status.get(filename, False) and timeout_counter < max_wait_time:
                progress += 2  # Increment progress by 2% each iteration
                if progress > 90:  # Cap progress at 90%
                    progress = 90
                
                logger.info(f"Waiting for final check completion for {filename}... Progress: {progress}%")
                yield f"data: {progress}\n\n"  # Send progress to the frontend
                
                # Wait but send keepalive messages during long waits
                for _ in range(5):  # Check every 0.5 seconds for 2.5 seconds total
                    time.sleep(0.5)
                    timeout_counter += 0.5
                    
                    # Send keepalive if we're still processing
                    now = time.time()
                    if now - last_keepalive > 15 and not final_check_status.get(filename, False):
                        yield f"data: {progress}\n\n"
                        last_keepalive = now
                        
                    if final_check_status.get(filename, False) or timeout_counter >= max_wait_time:
                        break
            
            # If we timed out waiting for completion
            if timeout_counter >= max_wait_time and not final_check_status.get(filename, False):
                yield f"data: Error: Processing is taking too long, please check server logs\n\n"
                return

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


def log_session_token_summary():
    """Log a complete summary of token usage and costs for the entire session."""
    if 'session_token_usage' in globals():
        # Calculate elapsed time
        elapsed_time = time.time() - session_token_usage.get('start_time', time.time())
        
        logger.info(f"\n")
        logger.info(f"============== COMPLETE SESSION TOKEN USAGE SUMMARY ==============")
        logger.info(f"Total Deepseek API calls made: {session_token_usage['api_calls']}")
        logger.info(f"Total tokens used: {session_token_usage['total_tokens']}")
        logger.info(f"  - Prompt tokens: {session_token_usage['prompt_tokens']}")
        logger.info(f"  - Completion tokens: {session_token_usage['completion_tokens']}")
        logger.info(f"Estimated total cost: ${session_token_usage['total_cost']:.4f}")
        logger.info(f"Session duration: {elapsed_time:.2f} seconds")
        logger.info(f"Cost per API call: ${session_token_usage['total_cost']/max(1, session_token_usage['api_calls']):.4f}")
        logger.info(f"Cost per token: ${session_token_usage['total_cost']/max(1, session_token_usage['total_tokens']):.6f}")
        logger.info(f"Tokens per second: {session_token_usage['total_tokens']/max(1, elapsed_time):.2f}")
        logger.info(f"==================================================================\n")
        
        # Reset for next session if needed
        # Uncomment the following line if you want to reset after logging
        # session_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'total_cost': 0.0, 'api_calls': 0, 'start_time': time.time()}



if __name__ == "__main__":
    # Load all disclosures from the custom file
    load_all_disclosures()
    
# Initialize BERT model
try:
    BERT_MODEL, BERT_TOKENIZER = get_bert_model()
    logger.info("BERT model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize BERT model: {e}")
    
    # Get the port from the environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
