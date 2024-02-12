from flask import Flask, render_template, request, redirect, url_for
import os
import json
import shutil

app = Flask(__name__)

# Path to the folders for classified emails
HAM_FOLDER = 'mails/ham/'
SPAM_FOLDER = 'mails/spam/'

# Ensure the existence of the ham and spam subfolders
os.makedirs(HAM_FOLDER, exist_ok=True)
os.makedirs(SPAM_FOLDER, exist_ok=True)

def read_emails_from_json():
    emails = []
    for filename in os.listdir('mails/'):
        if filename.endswith('.json'):
            with open('mails/'+filename, 'r') as file:
                json_file = json.load(file)
                json_file['filename'] = filename
                if 'is_spam' in json_file.keys():
                    json_file.pop('is_spam')
                emails.append(json_file)
    return emails

@app.route('/', methods=['GET'])
def index():
    emails = read_emails_from_json()
    return render_template('emails.html', emails=emails)

@app.route('/classify', methods=['POST'])
def classify_email():
    email_file = request.form['email_file']
    print(email_file)
    classification = request.form['classification']

    if classification == 'ham':
        shutil.move('mails/' + email_file, HAM_FOLDER + email_file)
    elif classification == 'spam':
        shutil.move('mails/' + email_file, SPAM_FOLDER + email_file)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
