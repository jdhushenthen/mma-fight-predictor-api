import pickle
from flask import Flask, Response
from flask_cors import CORS
import json
from datetime import datetime
from numpy import array

app = Flask(__name__)
CORS(app)

@app.route('/api/search/<string:fighter>')
def get_fighter_names(fighter):
    with open('data.json', 'r') as f:
        data = json.load(f)
        matching = [{"value": s, "label": s} for s in data.keys() if fighter.lower() in s.lower()]
    return {"options": matching}

@app.route('/api/stats/<string:fighter>')
def get_fighter_stats(fighter):
    with open('data.json', 'r') as f:
        data = json.load(f)
        stats = data[fighter]
    return stats

@app.route('/api/predict/<string:fighter1>/<string:fighter2>/<float:odds1>/<float:odds2>')
def get_current_time(fighter1, fighter2, odds1, odds2):
    with open('random_forest.pickle', 'rb') as f:
        model = pickle.load(f)

    with open('imputed_data.json', 'r') as f_dict:
        fighter_dict = json.load(f_dict)

    # convert decimal odds to probabilities
    odds1_probability = (1.0 / odds1) * 100.0
    odds2_probability = (1.0 / odds2) * 100.0

    # determine favourite and underdog
    if odds1_probability == odds2_probability:
        fighters_sorted = sorted([fighter1, fighter2], key=lambda s: s.lower())
        favourite = fighters_sorted[0]
        underdog = fighters_sorted[1]
    elif odds1_probability > odds2_probability:
        favourite = fighter1
        underdog = fighter2
    else:
        favourite = fighter2
        underdog = fighter1      

    # determine odds of favourite and underdog
    if favourite == fighter1:
        favourite_odds = odds1_probability
        underdog_odds = odds2_probability
    else:
        favourite_odds = odds2_probability
        underdog_odds = odds1_probability

    # get stats dictionary for each fighter from json file
    favourite_stats = fighter_dict[favourite]
    underdog_stats = fighter_dict[underdog]
    features_in = []

    # get lists of keys to iterate over
    favourite_keys = list(favourite_stats)
    underdog_keys = list(underdog_stats)

    # assign key values to list to pass to model, with special handling for age, inactivity and odds
    for i in range(len(favourite_keys)):
        if favourite_keys[i] == 'DOB':
            date1 = datetime.strptime(favourite_stats['DOB'], "%d-%m-%Y")
            date2 = datetime.strptime(underdog_stats['DOB'], "%d-%m-%Y")
            cur_date = datetime.now()

            favourite_feature = (cur_date - date1).days
            underdog_feature = (cur_date - date2).days

        elif favourite_keys[i] == 'Last Fight':
            date1 = datetime.strptime(favourite_stats['Last Fight'], "%m.%d.%Y")
            date2 = datetime.strptime(underdog_stats['Last Fight'], "%m.%d.%Y")
            cur_date = datetime.now()

            favourite_feature = (cur_date - date1).days
            underdog_feature = (cur_date - date2).days

        else:
            favourite_feature = favourite_stats[favourite_keys[i]]
            underdog_feature = underdog_stats[underdog_keys[i]]

        features_in.append(favourite_feature)
        features_in.append(underdog_feature)

    # insert decimal odds into list
    features_in.insert(12, favourite_odds)
    features_in.insert(12, underdog_odds)

    pred_list = [features_in]
    pred_list = array(pred_list, dtype=float)

    prediction = model.predict(pred_list)

    if int(prediction[0]):
        return_val = underdog
    else:
        return_val = favourite

    # returns the probability of the predicted winner
    prob = model.predict_proba(pred_list)[0][int(prediction[0])] * 100
    return {"winner": return_val, "probability": prob}
