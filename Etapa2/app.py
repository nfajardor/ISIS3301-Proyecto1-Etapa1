from flask import Flask, request, render_template
import pandas as pd
from joblib import load
from nltk.tokenize import TweetTokenizer


# Declare a Flask app
app = Flask(__name__)

# Función para tokenizar los comments
def tokenizer(text):
    tt = TweetTokenizer()
    return tt.tokenize(text)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = load("assets/modelo.joblib")
        
        # Get values through input bars
        text = request.form.get("text")
        print(text)
        # Put inputs to dataframe
        X = pd.DataFrame([[text]], columns = ["text"])
        
        # Get prediction

        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("front.html", output = prediction)
# Running the app
if __name__ == '__main__':
    app.run(debug = True)