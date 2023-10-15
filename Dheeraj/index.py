import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained logistic regression model from the pickle file
with open('your_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        values = [int(request.form['value1']), int(request.form['value2']), int(request.form['value3']),
                  int(request.form['value4']), int(request.form['value5']), int(request.form['value6']),
                  int(request.form['value7']), int(request.form['value8']), int(request.form['value9']),
                  int(request.form['value10']), int(request.form['value11'])]

        # Make a prediction using the model
        prediction = model.predict([values])

        return render_template('index.html', prediction=prediction[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
