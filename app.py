from flask import Flask, render_template, request, session, redirect
from flask_socketio import SocketIO, emit
import joblib
import os

app = Flask(__name__)
app.secret_key = "secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# Load trained files
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

messages = []

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/chat", methods=["POST"])
def chat():
    username = request.form.get("username")
    session["username"] = username
    return render_template("chat.html", username=username)

@socketio.on("send_message")
def handle_message(data):
    text = data["message"]
    username = session.get("username", "User")

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    if prediction == 1:
        result = "Toxic"
    else:
        result = "Non-Toxic"

    message_data = {
        "username": username,
        "message": text,
        "result": result
    }

    messages.append(message_data)
    emit("receive_message", message_data, broadcast=True)

@app.route("/admin")
def admin():
    return render_template("admin.html", messages=messages)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    socketio.run(app, host="0.0.0.0", port=port)