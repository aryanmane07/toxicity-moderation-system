from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
import joblib

app = Flask(__name__)
app.secret_key = "secretkey"
socketio = SocketIO(app)

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

messages = []


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/chat", methods=["POST"])
def chat():
    username = request.form.get("username")
    if not username:
        return redirect("/")
    session["username"] = username
    return render_template("chat.html", username=username)


@app.route("/admin")
def admin():
    return render_template("admin.html", messages=messages)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/clear")
def clear_chat():
    messages.clear()
    return redirect("/admin")


@socketio.on("send_message")
def handle_message(data):
    text = data["message"]
    username = session.get("username", "User")

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    if prediction == 0:
        label = "Non-Toxic"
    elif prediction == 1:
        label = "Mild"
    elif prediction == 2:
        label = "Moderate"
    else:
        label = "Severely Toxic"

    message_data = {
        "username": username,
        "message": text,
        "label": label
    }

    messages.append(message_data)

    emit("receive_message", message_data, broadcast=True)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    socketio.run(app, host="0.0.0.0", port=port)