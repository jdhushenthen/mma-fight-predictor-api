import pickle
from flask import Flask, Response
import json
from datetime import datetime
from numpy import array

app = Flask(__name__)

@app.route('/api/search/<string:fighter>')
def get_fighter_names(fighter):
    with open('data.json', 'r') as f:
        data = json.load(f)
        matching = [{"value": s, "label": s} for s in data.keys() if fighter in s]
    return {"options": matching}

@app.route('/api/stats/<string:fighter>')
def get_fighter_stats(fighter):
    with open('data.json', 'r') as f:
        data = json.load(f)
        stats = data[fighter]
    return stats

@app.route('/predict/<string:fighter1>/<string:fighter2>/<float:odds1>/<float:odds2>')
def get_current_time(fighter1, fighter2, odds1, odds2):
    with open('random_forest.pickle', 'rb') as f:
        model = pickle.load(f)

    with open('imputed_stats.json', 'r') as f_dict:
        fighter_dict = json.load(f_dict)

    # get stats dictionary for each fighter from json file
    fighter1_stats = fighter_dict[fighter1]
    fighter2_stats = fighter_dict[fighter2]
    features_in = []

    # get lists of keys to iterate over
    fighter1_keys = list(fighter1_stats)
    fighter2_keys = list(fighter2_stats)

    # assign key values to list to pass to model, with special handling for age, inactivity and odds
    for i in range(len(fighter1_keys)):
        if fighter1_keys[i] == 'DOB':
            date1 = datetime.strptime(fighter1_stats['DOB'], "%d-%m-%Y")
            date2 = datetime.strptime(fighter2_stats['DOB'], "%d-%m-%Y")
            cur_date = datetime.now()

            fighter1_feature = (cur_date - date1).days
            fighter2_feature = (cur_date - date2).days

        elif fighter1_keys[i] == 'Last Fight':
            date1 = datetime.strptime(fighter1_stats['Last Fight'], "%m.%d.%Y")
            date2 = datetime.strptime(fighter2_stats['Last Fight'], "%m.%d.%Y")
            cur_date = datetime.now()

            fighter1_feature = (cur_date - date1).days
            fighter2_feature = (cur_date - date2).days

        else:
            fighter1_feature = fighter1_stats[fighter1_keys[i]]
            fighter2_feature = fighter2_stats[fighter2_keys[i]]

        features_in.append(fighter1_feature)
        features_in.append(fighter2_feature)

    # insert decimal odds into list
    features_in.insert(12, odds1*10)
    features_in.insert(12, odds2*10)

    pred_list = [features_in]
    pred_list = array(pred_list, dtype=float)

    prediction = model.predict(pred_list)

    if int(prediction[0]):
        return_val = fighter2
    else:
        return_val = fighter1

    # returns the probability of the predicted winner
    prob = model.predict_proba(pred_list)[0][int(prediction[0])] * 100
    return {"winner": return_val, "probability": prob}


@app.route('/test/<string:test_val>')
def test_route(test_val):
    return {"test": test_val}
