import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
import pickle

from utils import onehotCategorical

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':


        # ========== Part 2.3 ==========
        # YOUR CODE START HERE

        #******NOTE_******** I was unableto one-hot encode the 'Store' feature in my ipython notebook model so
        #I did not use the onehotCategorical function on the 'store' user input_
        #NOTE_: I extracted user inputs and one-hot encoded in the same line
        # get request values
        # one-hot encode categorical variables
        store = request.form.get("store")
        store_type = onehotCategorical(int(request.form.get("store_type")), 4).astype(str)
        assortment = onehotCategorical(int(request.form.get("assortment")), 3).astype(str)
        state_holliday = onehotCategorical(int(request.form.get("state_holliday")), 4).astype(str)
        promo2 = request.form.get("promo2")
        promo = request.form.get("promo")
        day_of_the_week = request.form.get("day_of_the_week")
        month = request.form.get("month")
        school_holliday = request.form.get("school_holliday")

        # manually specify competition distance
        comp_dist = '5458.1'

        # build 1 observation for prediction
        entered_li = np.hstack([store, store_type, assortment, state_holliday, comp_dist, promo2, promo, day_of_the_week, month, school_holliday]).tolist()


        # ========== End of Part 2.3 ==========

        # make prediction
        prediction = model.predict(np.array(entered_li).reshape(1, -1))
        label = "$" + str(np.squeeze(prediction.round(2)))

        return render_template('index.html', label=label)

if __name__ == '__main__':
    # load ML model
    # ========== Part 2.2 ==========
    # YOUR CODE START HERE

    model = joblib.load('rm.pkl')

    # ========== End of Part 2.2 ==========
    # start API
    ##app.run(host='0.0.0.0', port=8000, debug=True)
    app.run(debug=True)
