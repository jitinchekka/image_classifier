from flask import Flask, render_template, request, redirect, url_for, session,jsonify
import util

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")
if __name__=="name":
    app.run(debug=True)