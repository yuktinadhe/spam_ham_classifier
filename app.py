from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("naive_bayes_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        ps = PorterStemmer()
        msg = re.sub('[^a-zA-Z]', ' ', message)
        msg = msg.lower()
        msg = msg.split()
        msg = [ps.stem(word) for word in msg if word not in stopwords.words('english')]
        final_msg = " ".join(msg)

        vect_msg = vectorizer.transform([final_msg])
        prediction = model.predict(vect_msg)

        result = "🚫 Spam" if prediction[0] == 1 else "✅ Ham"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
