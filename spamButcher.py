#!/usr/bin/env python

import requests
import imaplib
import email
from email.header import decode_header
import json
import sqlite3
import sys,os
import re
import openai

database_path = 'emails.db'

# IMAP server details
imap_url = '<imap.server.com>'
username = 'username'
password = 'userpassword'
full_name = 'your full name'
email_address = 'your email address'


openai.ap_key =  os.environ["OPENAI_API_KEY"]
# model_engine = "gpt-3.5-turbo"      # price: 0.003/0.006
# model_engine = "gpt-3.5-turbo-1106" # price: 0.001/0.002
model_engine = "gpt-3.5-turbo-0125" # price: 0.0005/0.0015
# model_engine = "gpt-4-1106-preview" # price: 0.01/0.03
# model_engine = "gpt-4"              # price: 0.03/0.06


def extract_is_spam_substring(s):
    # This pattern looks for any character sequence that starts with '{', followed by any
    # number of characters (non-greedily, as indicated by '?'), includes 'is_spam' : true,
    # and then ends with the first '}' encountered after 'is_spam'.
#    pattern = r'\{[^{]*"is_spam"\s*[^}]*\}'
    pattern = r'\{.*?"is_spam".*?\}'


    # Use re.search to find the first occurrence that matches the pattern
    match = re.search(pattern, s, re.DOTALL)
    
    if match:
        return match.group(0)  # Return the matched substring
    else:
        return None  # Or an appropriate response if not found

def chatOpenAI(query):
    msg = f"You are an AI trained to detect spam emails. Analyze the email content provided and determine if it is spam or not.\
You work for {full_name} ({email_address}).\
Respond with a JSON object indicating the spam status." + \
"For example, respond with '{\"is_spam\": true}' for spam emails and '{\"is_spam\": false}' for non-spam emails.\n\nMAIL:\n" + str(query)
    response = openai.chat.completions.create(
        model=model_engine,  # or any other available model
        messages=[
            # {"role": "system", "content": "Du redaksjonsjef for et e-handels selskap, som skal lage nye og bedre produkt titler til produktene på siden. De nye titlenes lengde har en hard limit på 40 tegn."},
            {"role": "user", "content": msg }
        ]
    )
    return response.choices[0].message.content



def chat(messages):
    r = requests.post(
        "http://0.0.0.0:11434/api/chat",
        json={"model": 'spamfilter', "messages": messages, "stream": True},
    )
    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
            # print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            return message

def is_spam(email_json):
    # response = chat([
    #     {
    #         "role": "user",
    #         "message": "Is the following email spam or not? Answer with ONLY '{\"is_spam\": true}' for spam emails and '{\"is_spam\": false}' for non-spam emails. ONLY THIS!\n\n"+ f"{email_json}",
    #     },
    # ])
    response = chatOpenAI(email_json)

    # Interpret the response to determine if it's spam
    # print (f"Response: {response}")
    _answer = response.lower().strip()
    # print(f"PRE-ANSWER: {_answer}")
    answer = extract_is_spam_substring(_answer)
    # print(f"ANSWER: {answer}")
    try:
        json_output = json.loads(answer)
        # print(f"JSON OUTPUT: {json_output}")
    except:
        # print(f"JSON ERROR: {answer}")
        json_output = {}
    # lines = answer.split('\n')
    # print(f"Lines found in answer: {len(lines)} : {lines}")
    return json_output

    #Check if datebase exists
if os.path.exists(database_path):
    print("Database exists, opening...")

    # Establish a connection to the SQLite database
    conn = sqlite3.connect('emails.db')
    c = conn.cursor()
else:
    print("Database does not exist. Creating...")
    # Establish a connection to the SQLite database
    conn = sqlite3.connect('emails.db')
    c = conn.cursor()

    # Create a table for storing emails and their spam status
    c.execute('''CREATE TABLE IF NOT EXISTS emails
                (id TEXT PRIMARY KEY, subject TEXT, date TEXT, from_email TEXT, to_email TEXT, body TEXT, is_spam BOOLEAN)''')
    print("Database created")


def main():

    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL(imap_url)
    mail.login(username, password)
    mail.select('inbox')  # Default to the inbox

    all_mails = []

    # Search for unread emails
    status, response = mail.search(None, '(UNSEEN)')
    if status == 'OK':
        unread_msg_nums = response[0].split()
        for e_id in unread_msg_nums:
            _, response = mail.fetch(e_id, '(BODY.PEEK[])')
            for part in response:
                if isinstance(part, tuple):
                    msg = email.message_from_bytes(part[1])
                    email_subject = decode_header(msg["subject"])[0][0]
                    email_date = msg["date"]
                    email_from = msg["from"]
                    email_to = msg["to"]
                    email_id = msg["message-id"]
                    if msg.is_multipart():
                        email_body = ''
                        for payload in msg.get_payload():
                            if payload.get_content_type() == 'text/plain':
                                email_body += payload.get_payload(decode=True).decode()
                    else:
                        email_body = msg.get_payload(decode=True).decode()

                    # Create a JSON object
                    email_details = {
                        "id": email_id,
                        "subject": str(email_subject),
                        "date": email_date,
                        "from": email_from,
                        "to": email_to,
                        "body": email_body
                    }
                    # Print and save the email details to a JSON file
                    c.execute("SELECT * FROM emails WHERE id=?", (email_details['id'],))
                    result = c.fetchone()

                    if result:
                        print(f"Email {email_details['id']} already exists in the database")
                        continue

                    print("RUNNING SPAM CHECKER...")
                    spam_status = is_spam(email_details)
                    retries = 0
                    while type(spam_status)==dict and (not 'is_spam' in spam_status.keys()) and retries < 5:
                        spam_status = is_spam(email_details)
                        print(f"RETRY {retries}: Is email spam: {spam_status}")
                        retries += 1

                    if (spam_status['is_spam']==True):
                        print("Mail is spam")
                    else:
                        print("Mail is not spam")

                    email_details['is_spam'] = spam_status['is_spam']

                    with open(f'mails/email_{e_id.decode()}.json', 'w') as json_file:
                        json.dump(email_details, json_file, indent=4)

                    c.execute("INSERT INTO emails VALUES (?, ?, ?, ?, ?, ?, ?)", 
                    (email_details['id'], email_details['subject'], email_details['date'], email_details['from'], 
                    email_details['to'], email_details['body'], email_details['is_spam']))
                    conn.commit()
                    all_mails.append(email_details)

                    #moving mails to 'SpamAI' folder
                    mail.copy(e_id, 'SpamAI')
                    print(f"Email {email_details['id']} moved to 'SpamAI' folder")
                    #deleting old mail
                    mail.store(e_id, '+FLAGS', '\\Deleted')
                    #print(f"Email {email_details['id']} deleted")

        mail.expunge()  # Clean up deleted emails
    # Logout
    mail.logout()
    conn.close()


if __name__ == '__main__':
    main()