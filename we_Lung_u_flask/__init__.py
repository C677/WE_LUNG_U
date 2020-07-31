from flask import Flask

app = Flask(__name__)

@app.route("/")
def we_lung_u():
    return "WE_LUNG_U"
