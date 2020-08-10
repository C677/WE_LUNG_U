#import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
#from flask_bootstrap import Bootstrap

app = Flask(__name__)
#Bootstrap(app)

@app.route('/')
def we_lung_u():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/check', methods = ['GET', 'POST'])
def check():
    if request.method == 'POST':
        # 파일이 없을 때
        if request.files['file'] is None:
            return render_template('check.html', title='Check', check_message_test='empty') 
        else:
            f = request.files['file']
            f.save('./static/img/'+secure_filename(f.filename))
            return render_template('check.html', title = 'Check', result = 'lung cancer', check_message_test=str(f.filename), LC=80, p=5, t=5, e=10)
    else:
        return render_template('check.html', title='Check', check_message_test='empty')

@app.route('/contact')
def contact():
    return render_template('contact.html', title = 'Contact')

@app.route('/team')
def team():
    return render_template('team.html')

if __name__ == "__main__":
  #  port = int(os.environ.get("PORT", 80))
    try:
        app.run(host="0.0.0.0", port=80, debug=True)
    except Exception as ex:
        print(ex)