from flask import Flask, render_template, request, redirect, url_for
import os
from model import predict_emotion
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_PATH = 'emotion_data.db'
conn = sqlite3.connect(DB_PATH)
conn.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                image_path TEXT,
                emotion TEXT,
                date TEXT
              )''')
conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    name = request.form['name']
    file = request.files['photo']

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        emotion = predict_emotion(file_path)
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO users (name, image_path, emotion, date) VALUES (?, ?, ?, ?)',
                     (name, file_path, emotion, date))
        conn.commit()
        conn.close()

        return render_template('result.html', name=name, emotion=emotion, image_path=file_path)
    return redirect(url_for('index'))

@app.route('/gallery')
def gallery():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, image_path, emotion, date FROM users ORDER BY id DESC")
    data = cur.fetchall()
    conn.close()
    return render_template('gallery.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
