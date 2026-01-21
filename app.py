from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import PyPDF2
import pathlib
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta
import re
from functools import lru_cache

# ---------------------- LOAD ENV ----------------------
env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

# GLOBAL STORAGE
college_data = ""
sessions = {}

# Initialize model once (reuse connection)
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 400,  # Reduced for faster responses
    }
)

# ---------------------- LOAD PDFs ----------------------

@lru_cache(maxsize=1)
def get_college_data():
    """Cache PDF data in memory"""
    global college_data
    if college_data:
        return college_data
    
    pdf_list = ["Updated college.pdf", "shift1.pdf", "shift2.pdf", "rr.pdf", "AI&DS TT-UPDATED.pdf"]
    data = ""
    
    for file in pdf_list:
        try:
            with open(file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        data += txt + "\n"
            print(f"✓ {file} loaded")
        except Exception as e:
            print(f"✗ Error loading {file}:", e)
    
    college_data = data
    return data

# Load PDFs on startup
college_data = get_college_data()

# ---------------------- TIMETABLE EXTRACTION ----------------------

@lru_cache(maxsize=8)
def extract_timetable(day_order, pdf_text):
    """Cache timetable extractions"""
    pattern = rf"DAY\s*ORDER\s*{day_order}.*?(?=DAY\s*ORDER\s*\d|$)"
    match = re.search(pattern, pdf_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(0).strip()
    
    return f"Timetable for Day Order {day_order} not found."

# ---------------------- DATE UTIL ----------------------

def get_day_order(date):
    weekday = date.weekday()
    if weekday == 6:
        return None
    
    OFFSET = -1
    day_order = ((weekday + OFFSET) % 6) + 1
    return day_order

# ---------------------- CHAT ROUTE ----------------------

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message", "").strip()
        session_id = data.get("sessionId", "default")
        
        if not message:
            return jsonify({"reply": "Please ask a question."}), 400
        
        if session_id not in sessions:
            sessions[session_id] = []
        
        # Limit session history to last 6 messages (3 exchanges)
        if len(sessions[session_id]) > 6:
            sessions[session_id] = sessions[session_id][-6:]
        
        sessions[session_id].append({"role": "user", "content": message})
        lower_msg = message.lower()
        
        # --------------------------------------------
        # QUICK RESPONSES (No LLM needed)
        # --------------------------------------------
        
        # Date/Time
        if any(x in lower_msg for x in ["date", "what day"]):
            now = datetime.now()
            reply = f"📅 Today is {now.strftime('%A, %B %d, %Y')}."
            sessions[session_id].append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})
        
        if any(x in lower_msg for x in ["time", "what time"]):
            now = datetime.now()
            reply = f"🕐 Current time is {now.strftime('%I:%M %p')}."
            sessions[session_id].append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})
        
        # Timetable
        if any(k in lower_msg for k in ["timetable", "schedule", "class", "period"]):
            today = datetime.now()
            
            if "tomorrow" in lower_msg:
                target = today + timedelta(days=1)
            elif "day after" in lower_msg:
                target = today + timedelta(days=2)
            else:
                target = today
            
            day_order = get_day_order(target)
            
            if day_order is None:
                reply = "📅 It's Sunday - No classes today!"
            else:
                timetable_text = extract_timetable(day_order, college_data)
                reply = f"📚 **Day Order {day_order}** Timetable:\n\n{timetable_text}"
            
            sessions[session_id].append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})
        
        # Day Order
        if "day order" in lower_msg:
            today = datetime.now()
            
            if "tomorrow" in lower_msg:
                target = today + timedelta(days=1)
            elif "day after" in lower_msg:
                target = today + timedelta(days=2)
            else:
                target = today
            
            day_order = get_day_order(target)
            
            if day_order is None:
                reply = "📅 It's Sunday - No day order today."
            else:
                reply = f"📋 Today is **Day Order {day_order}**."
            
            sessions[session_id].append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})
        
        # --------------------------------------------
        # LLM RESPONSE (Optimized)
        # --------------------------------------------
        
        # Build compact context
        recent_history = sessions[session_id][-4:] if len(sessions[session_id]) > 4 else sessions[session_id]
        
        prompt = f"""You are CampusGuide AI for The New College.

College Data:
{college_data[:3000]}

Recent Chat:
{json.dumps(recent_history[-2:])}

User: {message}

Instructions:
- Answer briefly and clearly
- Use college data for facts
- Keep responses under 150 words
- Use emojis sparingly
"""
        
        response = model.generate_content(prompt)
        reply = response.text
        
        sessions[session_id].append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})
    
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"reply": f"⚠️ Error: {str(e)}"}), 500

# ---------------------- RUN SERVER ----------------------
if __name__ == "__main__":
    print("🚀 Server starting...")
    app.run(port=4000, debug=False)  # debug=False for production