from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
import os
from dotenv import load_dotenv
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import uuid
from datetime import datetime
from supabase import create_client

load_dotenv()

app = Flask(__name__)
app.secret_key = 'TwgBCPnFezdZpQvOFjJ2x0sWaHuBl3pUwGUkoyV9heY'

# Initialize Supabase client
SUPABASE_URL = "https://xjnpaujygdxxgyczpxdz.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhqbnBhdWp5Z2R4eGd5Y3pweGR6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY1NTc0MjMsImV4cCI6MjA2MjEzMzQyM30.MABCkUhx6ZEsme8c-2sqH_6Y60kWZy1TD8MoUHHEph8")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# LangChain Gemini Wrapper
class LangchainGeminiWrapper:
    def __init__(self, model='gemini-1.5-flash', google_api_key=None):
        if not google_api_key:
            raise ValueError("Google API key is required for Gemini.")
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key)

    def generate(self, prompt):
        print(f"\n--- LLM Prompt Start ---\n{prompt}\n--- LLM Prompt End ---") # Debugging prompt
        generated_text = self.llm.invoke(prompt).content
        print(f"\n--- LLM Raw Output Start ---\n{generated_text}\n--- LLM Raw Output End ---") # Debugging output
        return {'results': [{'generated_text': generated_text}]}

# Initialize the Gemini LLM
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
llm_model = LangchainGeminiWrapper(google_api_key=gemini_api_key)

# --- Supabase Functions ---
def signup_user(email, password):
    try:
        existing_user = supabase.table('mockr').select('*').eq('email', email).execute()
        if existing_user.data:
            user_id = existing_user.data[0]['user_id']
            return user_id

        user_id = str(uuid.uuid4())
        supabase.table('mockr').insert({
            "email": email,
            "password": password,
            "user_id": user_id
        }).execute()

        return user_id

    except Exception as e:
        print(f"Signup error: {e}")
        return None

def login_user(email, password):
    try:
        response = supabase.table('mockr').select('*').eq('email', email).eq('password', password).execute()

        if response.data:
            user_id = response.data[0]['user_id']
            return user_id
        else:
            return None

    except Exception as e:
        print(f"Login error: {e}")
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
        interview_insert_response = supabase.table("interviews").insert(interview_data).execute()

        if interview_insert_response.data:
            print("Interview data inserted.")
        else:
            print(f"Failed to insert interview data: {interview_insert_response.error}")
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
            feedback_insert_response = supabase.table("feedback").insert(feedback_data).execute()
            if feedback_insert_response.data:
                print("Improvement feedback data inserted.")
            else:
                print(f"Failed to insert improvement feedback data: {feedback_insert_response.error}")
                print("Improvement feedback insert error details:", feedback_insert_response.error)

        print("All interview-related data saved successfully!")

    except Exception as e:
        print(f"Error saving interview data: {e}")
        print("General error during data saving:", e)

# --- PDF Processing and LLM Functions ---
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.strip() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    return text

def summarize_content(content):
    MAX_TOKENS = 4000
    if len(content.split()) > MAX_TOKENS:
        content = " ".join(content.split()[:MAX_TOKENS])

    prompt = f"Summarize this resume:\n{content}"

    response = llm_model.generate(prompt)
    return response.get('results', [{}])[0].get('generated_text', 'No summary generated')

def get_question_score_and_critique(question, answer, job_description, resume_summary, stage):
    """Evaluates a single answer and provides a score and critique."""
    prompt = (
        f"For the question: '{question}', the user answered: '{answer}'. "
        "Provide a score out of 10 and a brief critique. "
        "Format:\nSCORE: X/10\nCRITIQUE: Your critique.\n---END---"
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
            print("Could not parse question score.")

    critique_match = re.search(r"CRITIQUE:\s*(.*?)(?=\n---END---|$)", evaluation_text, re.IGNORECASE | re.DOTALL)
    if critique_match:
        critique = critique_match.group(1).strip()

    return score, critique

def generate_overall_feedback(qa_history, stage, llm_model_instance):
    """Generates overall interview feedback, focusing on areas for improvement."""
    full_qa_string = "\n".join([
        f"Q: {item['question']}\nA: {item['answer']}\nIndividual Score: {item['question_score']}/10\nCritique: {item['critique']}"
        for item in qa_history
    ])

    prompt = (
        f"Based on this transcript, provide an overall score, an improvements feedback score, and a summary of strengths and areas for improvement.\n"
        f"Transcript:\n{full_qa_string}\n\n"
        "Format:\nOVERALL INTERVIEW SCORE: X/100\nIMPROVEMENTS FEEDBACK SCORE: Y/10\nSUMMARY OF STRENGTHS AND AREAS FOR IMPROVEMENT:\nYour feedback.\n---END---"
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
            print("Could not parse overall interview score.")

    improvements_score_match = re.search(r"IMPROVEMENTS FEEDBACK SCORE:\s*(\d+)/10", feedback_text, re.IGNORECASE)
    if improvements_score_match:
        try:
            improvement_feedback_score = int(improvements_score_match.group(1))
        except ValueError:
            print("Could not parse improvements feedback score.")

    summary_and_improvements_match = re.search(r"SUMMARY OF STRENGTHS AND AREAS FOR IMPROVEMENT:\s*(.*?)(?=\n---END---|$)", feedback_text, re.IGNORECASE | re.DOTALL)
    if summary_and_improvements_match:
        improvement_comment = summary_and_improvements_match.group(1).strip()
        overall_summary_for_interviews_table = improvement_comment

    return overall_score, improvement_comment, improvement_feedback_score, overall_summary_for_interviews_table

# --- Agent and Task Classes ---
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
            f"Generate one interview question based on this context:\n{chr(10).join(full_context)}\n\n"
            f"Task: {task.description}\n"
            "Question:"
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

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_id = login_user(email, password)
        if user_id:
            session['user_id'] = user_id
            return redirect(url_for('interview_setup'))
        else:
            return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_id = signup_user(email, password)
        if user_id:
            session['user_id'] = user_id
            return redirect(url_for('interview_setup'))
        else:
            return render_template('signup.html', error="Signup failed")
    return render_template('signup.html')

@app.route('/interview_setup', methods=['GET', 'POST'])
def interview_setup():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        resume = request.files['resume']
        job_title = request.form['job_title']
        job_position = request.form['job_position']
        exp_level = request.form['exp_level']
        job_description = request.form['job_description']
        stage = request.form['stage']

        if resume:
            resume_text = extract_text_from_pdf(resume)
            resume_summary = summarize_content(resume_text)

            session['job_title'] = job_title
            session['job_position'] = job_position
            session['exp_level'] = exp_level
            session['job_description'] = job_description
            session['stage'] = stage
            session['resume_summary'] = resume_summary
            session['qa_history'] = []
            session['overall_score'] = 0
            session['improvement_feedback'] = ""

            return redirect(url_for('interview'))
        else:
            return render_template('interview_setup.html', error="Please upload your resume")

    return render_template('interview_setup.html')

@app.route('/interview')
def interview():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    job_title = session.get('job_title')
    job_position = session.get('job_position')
    exp_level = session.get('exp_level')
    job_description = session.get('job_description')
    stage = session.get('stage')
    resume_summary = session.get('resume_summary')
    qa_history = session.get('qa_history') or []

    # Initialize agents and tasks
    tech_agent = Agent(
        role="Technical Agent", goal="Ask relevant technical questions", backstory="An expert in technical domains, focused on assessing core skills.",
        allow_delegation=False, verbose=True, llm=llm_model
    )
    behav_agent = Agent(
        role="Behavioral Agent", goal="Ask questions related to behavioral skills and past experiences.", backstory="An HR specialist keen on understanding soft skills, teamwork, and problem-solving through past actions.",
        allow_delegation=False, verbose=True, llm=llm_model
    )
    hr_agent = Agent(
        role="HR Agent", goal="Evaluate cultural fit, motivations, and career aspirations.", backstory="An experienced HR recruiter focused on aligning candidate's values with company culture.",
        allow_delegation=False, verbose=True, llm=llm_model
    )

    task_dict = {
        "Technical": Task(description="Generate a single concise technical interview question.", agent=tech_agent, expected_output="A single technical question."),
        "Behavioral": Task(description="Generate a single concise behavioral interview question.", agent=behav_agent, expected_output="A single behavioral question."),
        "HR": Task(description="Generate a single concise HR interview question.", agent=hr_agent, expected_output="A single HR question.")
    }

    current_task = task_dict.get(stage)

    if current_task:
        full_context_for_llm = [
            f"Job Title: {job_title}",
            f"Job Position: {job_position}",
            f"Experience Level: {exp_level}",
            f"Job Description: {job_description}",
            f"Resume Summary: {resume_summary}",
            f"Previous Q&A History:\n{chr(10).join([f'Q: {q["question"]}\nA: {q["answer"]}' for q in qa_history])}"
        ]
        question = current_task.agent.perform_task(current_task, full_context_for_llm)
        session['current_question'] = question

        return render_template('interview.html', question=question)
    else:
        return "Invalid interview stage selected."


@app.route('/next_question')
def next_question():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if session.get('qa_history') and len(session.get('qa_history')) >= 5:
        return redirect(url_for('results'))

    job_title = session.get('job_title')
    job_position = session.get('job_position')
    exp_level = session.get('exp_level')
    job_description = session.get('job_description')
    stage = session.get('stage')
    resume_summary = session.get('resume_summary')
    qa_history = session.get('qa_history') or []

    # Initialize agents and tasks
    tech_agent = Agent(
        role="Technical Agent", goal="Ask relevant technical questions", backstory="An expert in technical domains, focused on assessing core skills.",
        allow_delegation=False, verbose=True, llm=llm_model
    )
    behav_agent = Agent(
        role="Behavioral Agent", goal="Ask questions related to behavioral skills and past experiences.", backstory="An HR specialist keen on understanding soft skills, teamwork, and problem-solving through past actions.",
        allow_delegation=False, verbose=True, llm=llm_model
    )
    hr_agent = Agent(
        role="HR Agent", goal="Evaluate cultural fit, motivations, and career aspirations.", backstory="An experienced HR recruiter focused on aligning candidate's values with company culture.",
        allow_delegation=False, verbose=True, llm=llm_model
    )

    task_dict = {
        "Technical": Task(description="Generate a single concise technical interview question.", agent=tech_agent, expected_output="A single technical question."),
        "Behavioral": Task(description="Generate a single concise behavioral interview question.", agent=behav_agent, expected_output="A single behavioral question."),
        "HR": Task(description="Generate a single concise HR interview question.", agent=hr_agent, expected_output="A single HR question.")
    }

    current_task = task_dict.get(stage)

    if current_task:
        full_context_for_llm = [
            f"Job Title: {job_title}",
            f"Job Position: {job_position}",
            f"Experience Level: {exp_level}",
            f"Job Description: {job_description}",
            f"Resume Summary: {resume_summary}",
            f"Previous Q&A History:\n{chr(10).join([f'Q: {q["question"]}\nA: {q["answer"]}' for q in qa_history])}"
        ]
        question = current_task.agent.perform_task(current_task, full_context_for_llm)
        session['current_question'] = question

        return redirect(url_for('interview'))
    else:
        return "Invalid interview stage selected."

@app.route('/end_interview')
def end_interview():
    session.clear()
    return redirect(url_for('index'))

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    answer = request.form['answer']

    question = session.get('current_question')
    job_description = session.get('job_description')
    resume_summary = session.get('resume_summary')
    stage = session.get('stage')

    question_score, answer_critique = get_question_score_and_critique(
        question,
        answer,
        job_description,
        resume_summary,
        stage
    )

    qa_history = session.get('qa_history') or []
    qa_history.append({
        "question": question,
        "answer": answer,
        "question_score": question_score,
        "critique": answer_critique
    })
    session['qa_history'] = qa_history

    session['current_question'] = None # Clear current question

    return redirect(url_for('next_question'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    qa_history = session.get('qa_history') or []
    return render_template('history.html', qa_history=qa_history)

@app.route('/results')
def results():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    qa_history = session.get('qa_history') or []
    stage = session.get('stage')

    overall_score, improvement_feedback_comment, improvement_feedback_score, llm_generated_overall_summary = \
        generate_overall_feedback(qa_history, stage, llm_model)

    session['overall_score'] = overall_score
    session['improvement_feedback'] = improvement_feedback_comment

    # Save interview data to Supabase
    user_id = session.get('user_id')
    job_title = session.get('job_title')
    job_position = session.get('job_position')
    exp_level = session.get('exp_level')
    job_description = session.get('job_description')

    insert_interview_data(
        user_id=user_id,
        job_title=job_title,
        category=stage,
        total_questions=len(qa_history),
        overall_score=overall_score,
        summary=llm_generated_overall_summary,
        qa_history=qa_history,
        improvement_feedback_comment=improvement_feedback_comment,
        improvement_feedback_score=improvement_feedback_score
    )

    return render_template('results.html', overall_score=overall_score, improvement_feedback=improvement_feedback_comment)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
