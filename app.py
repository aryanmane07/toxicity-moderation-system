from flask import Flask, render_template, request, redirect, session
from flask_socketio import SocketIO, emit
from datetime import datetime
import pickle
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

socketio = SocketIO(app, cors_allowed_origins="*")

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# In-memory storage
messages = []
banned_users = set()
severe_counts = {}

# Multi-class label mapping
label_map = {
    0: "Non-toxic",
    1: "Mild",
    2: "Moderate",
    3: "Severely Toxic"
}

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        session["username"] = username
        return redirect("/chat")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/chat")
def chat():
    if "username" not in session:
        return redirect("/login")
    return render_template(
        "chat.html",
        username=session["username"],
        messages=messages,
        total=len(messages)
    )

@app.route("/clear")
def clear_chat():
    messages.clear()
    return redirect("/chat")

@app.route("/admin")
def admin():
    return render_template(
        "admin.html",
        messages=messages,
        banned_users=banned_users,
        severe_counts=severe_counts,
        total=len(messages)
    )
# =========================
# SOCKET HANDLER
# =========================

@socketio.on("send_message")
def handle_message(data):

    if "username" not in session:
        return

    username = session["username"]
    text = data["text"]

    if username in banned_users:
        return

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    severity = label_map.get(prediction, "Non-toxic")

    if severity == "Severely Toxic":
        severe_counts[username] = severe_counts.get(username, 0) + 1
        if severe_counts[username] >= 5:
            banned_users.add(username)

    timestamp = datetime.now().strftime("%I:%M %p")

    message_data = {
        "username": username,
        "text": text,
        "severity": severity,
        "timestamp": timestamp
    }

    messages.append(message_data)

    emit("receive_message", {
        "message": message_data,
        "total": len(messages)
    }, broadcast=True)

# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)