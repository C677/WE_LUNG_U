import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def we_lung_u():
    return "WE_LUNG_U"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    try:
        app.run(host="0.0.0.0", port=80, debug=True)
    except Exception as ex:
        print(ex)