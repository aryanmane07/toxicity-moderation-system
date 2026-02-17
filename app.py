from flask import Flask, render_template, request, redirect, session
from flask_socketio import SocketIO, emit
from datetime import datetime

app = Flask(__name__)
app.secret_key = "secret123"

socketio = SocketIO(app)

messages = []
online_users = set()
banned_users = set()
toxic_counts = {}

# -------------------------------
# Simple Rule-Based Toxicity Model
# -------------------------------
def predict_toxicity(text):
    text = text.lower()

    severe_words = ["fuck", "die", "kill", "disappear"]
    mild_words = ["stupid", "idiot", "weird"]

    for word in severe_words:
        if word in text:
            return "Severely Toxic"

    for word in mild_words:
        if word in text:
            return "Mild"

    return "Non-toxic"

# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")

        if username in banned_users:
            return "You are banned."

        session["username"] = username
        return redirect("/chat")

    return render_template("login.html")


@app.route("/chat")
def chat():
    if "username" not in session:
        return redirect("/")

    return render_template(
        "chat.html",
        username=session["username"],
        messages=messages,
        total=len(messages),
        online=len(online_users)
    )


@app.route("/admin")
def admin():
    return render_template(
        "admin.html",
        total=len(messages),
        online=len(online_users),
        banned=banned_users
    )


@app.route("/clear")
def clear():
    messages.clear()
    toxic_counts.clear()
    return redirect("/chat")


@app.route("/logout")
def logout():
    user = session.get("username")
    online_users.discard(user)
    session.clear()
    return redirect("/")


# -------------------------------
# Socket Events
# -------------------------------

@socketio.on("connect")
def handle_connect():
    username = session.get("username")
    if username:
        online_users.add(username)
        emit("update_online", len(online_users), broadcast=True)


@socketio.on("disconnect")
def handle_disconnect():
    username = session.get("username")
    if username:
        online_users.discard(username)
        emit("update_online", len(online_users), broadcast=True)


@socketio.on("send_message")
def handle_message(data):
    username = session.get("username")

    if username in banned_users:
        emit("banned")
        return

    text = data.get("message")
    label = predict_toxicity(text)
    time = datetime.now().strftime("%I:%M %p")

    # Count only Severely Toxic messages
    if label == "Severely Toxic":
        toxic_counts[username] = toxic_counts.get(username, 0) + 1

        if toxic_counts[username] >= 5:
            banned_users.add(username)
            emit("banned")
            return

    message_data = {
        "user": username,
        "text": text,
        "label": label,
        "time": time
    }

    messages.append(message_data)

    emit("receive_message", message_data, broadcast=True)
    emit("update_total", len(messages), broadcast=True)


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5050, debug=True)