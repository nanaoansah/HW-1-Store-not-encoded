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
        #store_type = request.form.get("store_type")
        assortment = onehotCategorical(int(request.form.get("assortment")), 3).astype(str)
        #assortment = request.form.get("assortment")
        state_holliday = onehotCategorical(int(request.form.get("state_holliday")), 4).astype(str)
        #state_holliday = request.form.get("state_holliday")
        promo2 = request.form.get("promo2")
        promo = request.form.get("promo")
        day_of_the_week = request.form.get("day_of_the_week")
        month = request.form.get("month")
        school_holliday = request.form.get("school_holliday")

        # manually specify competition distance
        comp_dist = '5458.1'

        # build 1 observation for prediction
        entered_li = np.hstack([store, store_type, assortment, state_holliday, comp_dist, promo2, promo, day_of_the_week, month, school_holliday]).tolist()

        #entered_li length = 18
        #entered_li = [store, '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', comp_dist, promo2, promo, day_of_the_week, month, school_holliday]
        #entered_li = np.hstack([store, store_type, [0, 0, 0, 0, 0, 0, 0], comp_dist, promo2, promo, day_of_the_week, month, school_holliday]).tolist()
        #if store_type == '1':
        #    entered_li[1] = '1'
        #if store_type == '2':
        #    entered_li[2] = '1'
        #if store_type == '3':
        #    entered_li[3] = '1'
        #if store_type == '4':
        #    entered_li[4] = '1'
        #if assortment == '1':
        #    entered_li[5] = '1'
        #if assortment == '2':
        #    entered_li[6] = '1'
        #if assortment == '3':
        #    entered_li[7] = '1'
        #if state_holliday == '1':
        #    entered_li[8] = '1'
        #if state_holliday == '2':
        #    entered_li[9] = '1'
        #if state_holliday == '3':
        #    entered_li[10] = '1'
        #if state_holliday == '4':
        #    entered_li[11] = '1'

        #new_entered_li = list(np.float_(entered_li))
        #test_in = ','.join(entered_li)








        # ========== End of Part 2.3 ==========

        # make prediction
        #print(model)
        prediction = model.predict(np.array(entered_li).reshape(1, -1))
        label = "$" + str(np.squeeze(prediction.round(2)))
        ##label = test_in

        return render_template('index.html', label=label)

if __name__ == '__main__':
    # load ML model
    # ========== Part 2.2 ==========
    # YOUR CODE START HERE
    ##with open('rm.pkl', 'rb') as f:
        ##model = pickle.load(f)

    model = joblib.load('rm.pkl')

    # ========== End of Part 2.2 ==========
    # start API
    ##app.run(host='0.0.0.0', port=8000, debug=True)
    app.run(debug=True)
