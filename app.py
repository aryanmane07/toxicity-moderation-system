import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
import pickle
import os

# ===============================
# FLASK SETUP
# ===============================

app = Flask(__name__)
app.secret_key = "supersecretkey"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet"
)

# ===============================
# LOAD MODEL
# ===============================

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ===============================
# LABEL MAP
# ===============================

label_map = {
    0: "Non Toxic",
    1: "Mild",
    2: "Moderate",
    3: "Severely Toxic"
}

# ===============================
# MEMORY STORAGE
# ===============================

messages = []
banned_users = {}
user_message_count = {}

# ===============================
# ROUTES
# ===============================

@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        session["username"] = username
        return redirect(url_for("chat"))
    return render_template("login.html")

@app.route("/chat")
def chat():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", username=session["username"])

@app.route("/admin")
def admin():
    return render_template("admin.html", banned_users=banned_users)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ===============================
# SOCKET EVENTS
# ===============================

@socketio.on("send_message")
def handle_message(data):
    username = session.get("username")

    if not username:
        return

    # Check if banned
    if username in banned_users:
        emit("banned", {"message": "You are banned."})
        return

    text = data["message"]

    # Predict toxicity
    vect = vectorizer.transform([text])
    prediction = model.predict(vect)[0]
    label = label_map.get(prediction)

    # Track severely toxic messages
    if username not in user_message_count:
        user_message_count[username] = 0

    if label == "Severely Toxic":
        user_message_count[username] += 1

    # Ban after 5 severely toxic messages
    if user_message_count[username] >= 5:
        banned_users[username] = "Banned for toxicity"
        emit("banned", {"message": "You are banned for severe toxicity."})
        return

    message_data = {
        "username": username,
        "message": text,
        "label": label
    }

    messages.append(message_data)

    socketio.emit("receive_message", message_data)

# ===============================
# RUN SERVER (Railway Compatible)
# ===============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    socketio.run(app, host="0.0.0.0", port=port)