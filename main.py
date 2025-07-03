import os
import time
import json
import threading
import supabase
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from flask_socketio import SocketIO, emit
from supabase import create_client
from datetime import datetime
import uuid
import random
import re
import azure.cognitiveservices.speech as speechsdk
import io
import tempfile
import base64  # Import base64 for handling audio chunks
from threading import Thread, Event
import pdfplumber

# LangChain and Ollama
from langchain_ollama import OllamaLLM

# --- Global variables for real-time transcription, cumulative interview transcript, and analysis ---
# This will hold the latest recognized chunk for immediate real-time display (frontend uses this for 'partial_transcript')
current_display_transcript_chunk = ""

# This will accumulate the complete Q&A history of the entire interview for final LLM processing
cumulative_interview_transcript = ""

# Cumulative analysis results for the entire interview
interview_analysis_metrics = {
    "word_count": 0,
    "filler_words_count": 0,
    "grammar_suggestions": []
}

# Define common filler words for analysis
FILLER_WORDS = ["um", "uh", "like", "you know", "so", "basically", "actually", "right", "okay"]

app = Flask(__name__)
app.secret_key = 'TwgBCPnFezdZpQvOFjJ2x0sWaHuBl3pUwGUkoyV9heY'  # !!! IMPORTANT: Change this for production !!!
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True # Keep this for security
app.config['SESSION_COOKIE_SECURE'] = False # Set to False for local HTTP development
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for simplicity in dev

# --- Initialize Supabase client ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://xjnpaujygdxxgyczpxdz.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhqbnBhdWp5Z2R4eGd5Y3pweGR6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY1NTc0MjMsImV4cCI6MjA2MjEzMzQyM30.MABCkUhx6ZEsme8c-2sqH_6Y60kWZy1TD8MoUHHEph8")

print(f"Supabase URL: {SUPABASE_URL}")
print(f"Supabase Key: {SUPABASE_KEY}")

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Azure Speech Configuration ---
AZURE_SUBSCRIPTION_KEY = os.environ.get("AZURE_SPEECH_KEY", "E18UvRZ7iPfStToXh8a095hADvxQt3hz3vxwwejxnPaPK0czyJQQJ99BDAC4f1cMXJ3w3AAAYACOGW0nN")
AZURE_REGION = os.environ.get("AZURE_SPEECH_REGION", "westus")

print(f"Azure Subscription Key: {AZURE_SUBSCRIPTION_KEY}")
print(f"Azure Region: {AZURE_REGION}")

# --- LangChain Ollama Wrapper ---
class LangchainOllamaWrapper:
    def __init__(self, model='llama2', base_url='http://localhost:11434'):
        self.llm = OllamaLLM(model=model, base_url=base_url)

    def generate(self, prompt):
        print(f"\n--- LLM Prompt Start ---\n{prompt}\n--- LLM Prompt End ---")
        try:
            generated_text = self.llm.invoke(prompt)
            print(f"\n--- LLM Raw Output Start ---\n{generated_text}\n--- LLM Raw Output End ---")
            return {'results': [{'generated_text': generated_text}]}
        except Exception as e:
            print(f"Error calling Ollama LLM: {e}")
            return {'results': [{'generated_text': 'Error: Could not generate response from LLM.'}]}


# Initialize the Ollama LLM
llm_model = LangchainOllamaWrapper()

# --- Supabase Functions (Unchanged) ---
def signup_user(email, password):
    try:
        existing_user = supabase_client.table('mockr').select('*').eq('email', email).execute()
        if existing_user.data:
            user_id = existing_user.data[0]['user_id']
            flash(f"User already exists. User ID: {user_id}", "info")
            return user_id

        user_id = str(uuid.uuid4())
        supabase_client.table('mockr').insert({
            "email": email,
            "password": password,
            "user_id": user_id
        }).execute()
        flash(f"User signed up. User ID: {user_id}", "success")
        return user_id
    except Exception as e:
        flash(f"Signup error: {e}", "error")
        return None

def login_user(email, password):
    try:
        response = supabase_client.table('mockr').select('*').eq('email', email).eq('password', password).execute()
        if response.data:
            user_id = response.data[0]['user_id']
            flash("Login successful!", "success")
            return user_id
        else:
            flash("Invalid email or password.", "error")
            return None
    except Exception as e:
        flash(f"Login error: {e}", "error")
        return None

def insert_interview_data(
    user_id: str,
    job_title: str,
    category: str,
    total_questions: int,
    overall_score: int,
    summary: str,
    qa_history: list,
    improvement_feedback_comment: str,
    improvement_feedback_score: int = 0
):
    try:
        now = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"
        interview_id = str(uuid.uuid4())

        interview_data = {
            "interview_id": interview_id,
            "user_id": user_id,
            "job_title": job_title,
            "category": category,
            "total_questions": total_questions,
            "score": overall_score,
            "summary": summary,
            "qa_history": qa_history,
            "started_at": now,
            "ended_at": now,
            "is_completed": True
        }
        print("Inserting into 'interviews' payload:", interview_data)
        interview_insert_response = supabase_client.table("interviews").insert(interview_data).execute()

        if interview_insert_response.data:
            flash("Interview data inserted.", "success")
        else:
            flash(f"Failed to insert interview data: {interview_insert_response.error}", "error")
            print("Interview insert error details:", interview_insert_response.error)

        if improvement_feedback_comment:
            feedback_data = {
                "id": str(uuid.uuid4()),
                "interview_id": interview_id,
                "category": "Improvements",
                "marks": improvement_feedback_score,
                "comment": improvement_feedback_comment,
                "created_at": now
            }
            print("Inserting into 'feedback' payload:", feedback_data)
            feedback_insert_response = supabase_client.table("feedback").insert(feedback_data).execute()
            if feedback_insert_response.data:
                flash("Improvement feedback data inserted.", "success")
            else:
                flash(f"Failed to insert improvement feedback data: {feedback_insert_response.error}", "error")
                print("Interview insert error details:", feedback_insert_response.error)

        flash("All interview-related data saved successfully!", "success")
    except Exception as e:
        flash(f"Error saving interview data: {e}", "error")
        print("General error during data saving:", e)

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.strip() + "\n"
    except Exception as e:
        # Flash the error to the user and print it to the console
        flash(f"Error extracting text from PDF: {e}", "error")
        print(f"Error during PDF text extraction: {e}")
        return ""
    return text


def summarize_content(content):
    MAX_TOKENS = 4000
    if len(content.split()) > MAX_TOKENS:
        content = " ".join(content.split()[:MAX_TOKENS])

    prompt = (
        "You are an AI assistant. Given the full text of a resume, write a concise and informative summary "
        "highlighting the candidate's key skills, experiences, education, and achievements.\n\n"
        f"Resume:\n{content}\n\nSummary:"
    )
    response = llm_model.generate(prompt)
    return response.get('results', [{}])[0].get('generated_text', 'No summary generated')

def get_question_score_and_critique(question, answer, job_description, resume_summary, stage):
    prompt = (
        f"You are an AI interviewer evaluating a candidate's answer to a specific question for a {stage} interview. "
        f"The job description is: {job_description}. The candidate's resume summary is: {resume_summary}. "
        f"Question: {question}\nCandidate's Answer: {answer}\n\n"
        "Provide a score for this answer out of 10. Also, provide a brief critique of the answer, "
        "highlighting strengths and areas for improvement for THIS specific answer. "
        "Strictly adhere to the following format, including the '---END---' marker:\n"
        "SCORE: X/10\n"
        "CRITIQUE: Your detailed critique here.\n"
        "---END---"
    )
    response = llm_model.generate(prompt)
    evaluation_text = response.get('results', [{}])[0].get('generated_text', 'No evaluation generated.')

    print(f"\n--- Raw LLM Output for Question Evaluation ---\n{evaluation_text}\n--- End Raw LLM Output ---")

    score = 0
    critique = "No critique provided."

    score_match = re.search(r"SCORE:\s*(\d+)/10", evaluation_text, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
        except ValueError:
            flash("Could not parse question score.", "warning")

    critique_match = re.search(r"CRITIQUE:\s*(.*?)(?=\n---END---|$)", evaluation_text, re.IGNORECASE | re.DOTALL)
    if critique_match:
        critique = critique_match.group(1).strip()

    return score, critique

def generate_overall_feedback(qa_history, stage, llm_model_instance):
    full_qa_string = "\n".join([
        f"Q: {item['question']}\nA: {item['answer']}\nIndividual Score: {item['question_score']}/10\nCritique: {item['critique']}"
        for item in qa_history
    ])

    prompt = (
        "You are an AI interview evaluator. Based on the candidate's entire interview performance "
        "from the provided questions, answers, and individual scores, "
        "provide a concise summary of their overall strengths and, most importantly, "
        "**specific, actionable areas for improvement**. "
        "Also, give an overall interview score out of 100 and a numerical score out of 10 for the quality/helpfulness of the 'Improvements' feedback itself.\n\n"
        "Strictly adhere to the following format, including the '---END---' marker:\n"
        "OVERALL INTERVIEW SCORE: X/100\n"
        "IMPROVEMENTS FEEDBACK SCORE: Y/10\n"
        "SUMMARY OF STRENGTHS AND AREAS FOR IMPROVEMENT:\nYour detailed improvement feedback here.\n"
        "---END---"
        f"\n\nCandidate's Interview Transcript (Interview Stage: {stage}):\n{full_qa_string}\n\n"
        "Now, generate the feedback in the specified format:"
    )

    response = llm_model_instance.generate(prompt)
    feedback_text = response.get('results', [{}])[0].get('generated_text', 'No feedback generated.')

    print(f"\n--- Raw LLM Output for Overall Feedback ---\n{feedback_text}\n--- End Raw LLM Output ---")

    overall_score = 0
    improvement_feedback_score = 0
    improvement_comment = "No improvement feedback generated."
    overall_summary_for_interviews_table = "No overall summary generated."

    overall_match = re.search(r"OVERALL INTERVIEW SCORE:\s*(\d+)/100", feedback_text, re.IGNORECASE)
    if overall_match:
        try:
            overall_score = int(overall_match.group(1))
        except ValueError:
            flash("Could not parse overall interview score.", "warning")

    improvements_score_match = re.search(r"IMPROVEMENTS FEEDBACK SCORE:\s*(\d+)/10", feedback_text, re.IGNORECASE)
    if improvements_score_match:
        try:
            improvement_feedback_score = int(improvements_score_match.group(1))
        except ValueError:
            flash("Could not parse improvements feedback score.", "warning")

    summary_and_improvements_match = re.search(r"SUMMARY OF STRENGTHS AND AREAS FOR IMPROVEMENT:\s*(.*?)(?=\n---END---|$)", feedback_text, re.IGNORECASE | re.DOTALL)
    if summary_and_improvements_match:
        # !!! CORRECTED TYPO HERE !!!
        improvement_comment = summary_and_improvements_match.group(1).strip()
        overall_summary_for_interviews_table = improvement_comment

    return overall_score, improvement_comment, improvement_feedback_score, overall_summary_for_interviews_table

# --- Agent and Task Classes (Unchanged) ---
class Agent:
    def __init__(self, role, goal, backstory, allow_delegation, verbose, llm):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.allow_delegation = allow_delegation
        self.verbose = verbose
        self.llm = llm

    def perform_task(self, task, full_context):
        prompt = (
            f"You are an interview agent with the role: {self.role}. "
            f"Your goal is: {self.goal}. "
            f"Your backstory: {self.backstory}. "
            "Based on the candidate's resume summary, job title, position, experience level, "
            "job description, and previous questions/answers, generate *only one* concise and relevant interview question. "
            "Do NOT include any conversational filler, greetings, or explanations. "
            "Start directly with the question.\n\n"
            f"Context:\n{chr(10).join(full_context)}\n\n"
            f"Task: {task.description}\nExpected Output: {task.expected_output}\n"
            "Generate ONLY the interview question here:"
        )
        response = self.llm.generate(prompt)
        question = response.get('results', [{}])[0].get('generated_text', 'Task not performed.').strip()
        question = re.sub(r"^(Question:\s*|^\d+\.\s*)", "", question, flags=re.MULTILINE).strip()
        return question

class Task:
    def __init__(self, description, agent, expected_output):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output

# --- Azure Speech Functions (Modified for Streaming and TTS) ---

# Global dictionary to hold PushAudioInputStreams and Recognizer instances per session
active_recognizers = {}

def start_stt_stream(sid):
    global current_display_transcript_chunk, cumulative_interview_transcript, interview_analysis_metrics
    if not AZURE_SUBSCRIPTION_KEY or not AZURE_REGION:
        emit('stt_error', {'message': "Azure Speech Service keys are not configured."}, room=sid)
        return

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SUBSCRIPTION_KEY, region=AZURE_REGION)
    speech_config.speech_recognition_language = "en-US"
    
    # Using a push stream requires a specific audio config
    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    final_transcript_for_current_answer = []
    
    def recognizing_handler(evt):
        global current_display_transcript_chunk
        current_display_transcript_chunk = evt.result.text
        emit('partial_transcript', {'text': current_display_transcript_chunk}, room=sid)

    def recognized_handler(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if evt.result.text: # Only append if there's text
                final_transcript_for_current_answer.append(evt.result.text)
                emit('final_transcript_segment', {'text': evt.result.text}, room=sid)
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            emit('stt_warning', {'message': "No speech recognized in this segment."}, room=sid)
    
    def canceled_handler(evt):
        print(f"CANCELED: Reason={evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"CANCELED: ErrorDetails={evt.error_details}")
            emit('stt_error', {'message': f"Speech recognition canceled: {evt.error_details}"}, room=sid)

    def session_stopped_handler(evt):
        print(f"Session stopped for SID: {sid}")
        # When session stops, we want to stop recognition gracefully
        if sid in active_recognizers:
            active_recognizers[sid]['stop_listening_flag'] = True

    recognizer.recognizing.connect(recognizing_handler)
    recognizer.recognized.connect(recognized_handler)
    recognizer.canceled.connect(canceled_handler)
    recognizer.session_stopped.connect(session_stopped_handler)

    active_recognizers[sid] = {
        'recognizer': recognizer,
        'push_stream': push_stream,
        'final_transcript_for_current_answer': final_transcript_for_current_answer,
        'stop_listening_flag': False,
        'recognition_thread': None
    }
    
    def recognize_loop():
        # This will block until recognition is stopped
        recognizer.start_continuous_recognition()
        while not active_recognizers.get(sid, {}).get('stop_listening_flag', True):
            time.sleep(0.1)
        recognizer.stop_continuous_recognition()
        print(f"Continuous recognition stopped for SID: {sid}")

    # Start the recognition loop in a background thread
    recognition_thread = Thread(target=recognize_loop)
    active_recognizers[sid]['recognition_thread'] = recognition_thread
    recognition_thread.start()
    
    emit('stt_ready', {'message': 'Ready to receive audio!'}, room=sid)


def stop_stt_stream(sid):
    if sid in active_recognizers:
        print(f"Attempting to stop STT stream for SID: {sid}")
        recognizer_info = active_recognizers[sid]
        recognizer_info['stop_listening_flag'] = True
        
        # Closing the push_stream signals the end of audio data to the recognizer
        recognizer_info['push_stream'].close()
        
        # Wait for the recognition thread to finish
        if recognizer_info['recognition_thread']:
             recognizer_info['recognition_thread'].join(timeout=2.0)

        del active_recognizers[sid]
        print(f"STT stream stopped and cleaned up for SID: {sid}")


def get_final_transcription_from_stream(sid):
    if sid in active_recognizers:
        return " ".join(active_recognizers[sid]['final_transcript_for_current_answer'])
    return ""


def clear_transcription_buffer(sid):
    global current_display_transcript_chunk
    current_display_transcript_chunk = ""
    if sid in active_recognizers:
        active_recognizers[sid]['final_transcript_for_current_answer'].clear()
        print(f"Transcription buffer cleared for SID: {sid}")


def text_to_speech_to_base64(text):
    if not AZURE_SUBSCRIPTION_KEY or not AZURE_REGION:
        flash("Azure Speech Service keys are not configured for Text-to-Speech.", "error")
        print("Azure TTS Error: Subscription key or region missing.")
        return None

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SUBSCRIPTION_KEY, region=AZURE_REGION)
    speech_config.speech_synthesis_voice_name = 'en-US-AriaNeural'
    speech_config.speech_synthesis_output_format = speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    try:
        result = speech_synthesizer.speak_text_async(text).get()
    except Exception as e:
        flash(f"Error during Azure TTS synthesis: {e}", "error")
        print(f"Azure TTS Synthesis Exception: {e}")
        return None

    if result.reason == speechsdk.SpeechReason.SynthesizingAudioCompleted:
        return base64.b64encode(result.audio_data).decode('utf-8')
    elif result.reason == speechsdk.SpeechReason.Canceled:
        cancellation_details = result.cancellation_details
        error_message = f"Speech synthesis canceled: {cancellation_details.reason}"
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                error_message += f" | Error details: {cancellation_details.error_details}"
        flash(error_message, "error")
        print(error_message)
    return None

@app.route("/", methods=["GET", "POST"])
def login_signup():
    if "user_id" in session:
        return redirect(url_for("interview_setup"))

    if request.method == "POST":
        auth_option = request.form.get("auth_option")
        email = request.form.get("email")
        password = request.form.get("password")
        
        user_id = None
        if auth_option == "signup":
            user_id = signup_user(email, password)
        elif auth_option == "login":
            user_id = login_user(email, password)

        if user_id:
            session["user_id"] = user_id
            session["interview_data"] = {}
            return redirect(url_for("interview_setup"))
        else:
            return render_template("login.html")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login_signup"))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# ==============================================================================
# === CORRECTED AND IMPROVED interview_setup ROUTE =============================
# ==============================================================================
@app.route("/interview_setup", methods=["GET", "POST"])
def interview_setup():
    if "user_id" not in session:
        flash("Please log in to set up an interview first.", "warning")
        return redirect(url_for("login_signup"))
    
    if "interview_data" not in session:
        session["interview_data"] = {}

    print("User ID found in session:", session.get("user_id"))
    print("Initial interview data in session:", session.get("interview_data"))
    
    if request.method == "POST":
        job_title = request.form.get("job_title")
        job_description = request.form.get("job_description")
        interview_stage = request.form.get("interview_stage")
        exp_level = request.form.get("experience_level")
        uploaded_pdf = request.files.get("resume_pdf")

        # --- DEBUGGING PRINTS FOR FORM DATA ---
        print(f"DEBUG FORM DATA: Job Title: '{job_title}'")
        print(f"DEBUG FORM DATA: Job Description: '{job_description}'")
        print(f"DEBUG FORM DATA: Interview Stage: '{interview_stage}'")
        print(f"DEBUG FORM DATA: Experience Level: '{exp_level}'")
        print(f"DEBUG FORM DATA: Uploaded PDF object: {uploaded_pdf}")
        if uploaded_pdf:
            print(f"DEBUG FORM DATA: Uploaded PDF filename: '{uploaded_pdf.filename}'")
            print(f"DEBUG FORM DATA: Uploaded PDF content_type: '{uploaded_pdf.content_type}'")
        else:
            print("DEBUG FORM DATA: No 'resume_pdf' found in request.files.")
        # --- END DEBUGGING PRINTS ---

        # Original validation block
        if not all([job_title, job_description, interview_stage, exp_level, uploaded_pdf and uploaded_pdf.filename]):
            flash("Please complete all fields and ensure a resume file is uploaded.", "error")
            print("DEBUG: Validation failed - one or more required fields are missing or PDF is not uploaded.")
            return render_template("interview_setup.html")

        print("All form fields are present. Proceeding to PDF processing...") # Added this print

        pdf_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                uploaded_pdf.save(tmp_pdf.name)
                pdf_file_path = tmp_pdf.name
            print(f"DEBUG: Saved uploaded PDF to temporary path: {pdf_file_path}")

            resume_text = extract_text_from_pdf(pdf_file_path)
            
            # The more detailed text display is now handled within extract_text_from_pdf itself
            # This simplified print just confirms it's not empty from this level.
            print(f"DEBUG: Length of final extracted text from interview_setup: {len(resume_text.strip())}")

            if not resume_text.strip():
                print("DEBUG: Resume text is empty or only whitespace after extraction (from interview_setup).")
                flash("Could not extract any meaningful text from the uploaded PDF. Please ensure it's a text-based (not scanned) PDF or try another file.", "error")
                return render_template("interview_setup.html")

            print("Summarizing resume text...")
            resume_summary = summarize_content(resume_text)
            
            print("--- Generated Resume Summary ---")
            print(resume_summary)
            print("---------------------------------")
            
            if not resume_summary or "No summary generated" in resume_summary or "Error" in resume_summary:
                flash("Failed to summarize the resume. This could be an issue with the LLM connection. Please try again.", "error")
                print("DEBUG: Resume summary failed or was an error message.")
                return render_template("interview_setup.html")

            flash("Resume summarized successfully! Starting the interview.", "success")

            session["interview_data"] = {
                "job_title": job_title,
                "job_position": job_description,
                "exp_level": exp_level,
                "job_description": job_description,
                "interview_stage": interview_stage,
                "resume_summary": resume_summary,
                "qa_history": [],
                "answered_questions": 0,
                "current_question": None,
                "overall_interview_score": 0,
                "improvement_feedback_comment": "",
                "improvement_feedback_score": 0,
                "llm_generated_overall_summary": "",
                "interview_running": False
            }
            session.modified = True
            
            print("Session data populated. Redirecting to /interview...")
            return redirect(url_for("interview"))

        except Exception as e:
            print(f"ERROR: An unexpected error occurred during PDF processing: {e}")
            import traceback # Added for more detailed error in console
            traceback.print_exc()
            flash(f"An unexpected error occurred while processing your resume: {e}", "error")
            return render_template("interview_setup.html")

        finally:
            if pdf_file_path and os.path.exists(pdf_file_path):
                os.unlink(pdf_file_path)
                print(f"DEBUG: Cleaned up temporary PDF file: {pdf_file_path}")
    
    return render_template("interview_setup.html")
@app.route("/interview")
def interview():
    if "user_id" not in session or not session.get("interview_data"):
        flash("Please set up an interview first.", "warning")
        return redirect(url_for("interview_setup"))

    interview_data = session["interview_data"]
    question_audio_base64 = None
    if interview_data.get("current_question"):
        question_audio_base64 = text_to_speech_to_base64(interview_data["current_question"])

    return render_template("interview.html", interview_data=interview_data, question_audio_base64=question_audio_base64)


# --- WebSocket Event Handlers ---

@socketio.on('connect')
def test_connect():
    sid = request.sid
    print(f"Client connected: {sid}")
    start_stt_stream(sid)

@socketio.on('disconnect')
def test_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    stop_stt_stream(sid)

@socketio.on('send_audio_chunk')
def handle_audio_chunk(data):
    sid = request.sid
    if sid in active_recognizers:
        # Audio chunks from JS are raw PCM data, not base64
        audio_chunk = data['audio_chunk']
        active_recognizers[sid]['push_stream'].write(audio_chunk)
    else:
        print(f"No active recognizer for SID {sid}. Audio chunk dropped.")


@socketio.on('end_audio_stream')
def handle_end_audio_stream():
    """
    Called when the client stops sending audio for the current user answer.
    We retrieve the complete recognized answer for the current turn.
    """
    sid = request.sid
    if sid in active_recognizers:
        print(f"Audio stream ended by client for SID: {sid}. Waiting for final transcription...")
        
        # Give Azure a moment to process the last bit of audio
        time.sleep(1.0) 

        final_user_answer = get_final_transcription_from_stream(sid)
        print(f"Final Answer for SID {sid}: {final_user_answer}")
        
        emit('full_transcript_ready', {'text': final_user_answer}, room=sid)

        # Buffer should be cleared after the answer is processed,
        # typically before the next question is asked.
        clear_transcription_buffer(sid)
    else:
        print(f"Received end_audio_stream but no active recognizer for SID: {sid}")

if __name__ == '__main__':
    socketio.run(app, debug=True)