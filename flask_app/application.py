import random
from flask import Flask, render_template, request
from recommender import sql_data, user_input_vector, recommender_build, custom_recommendation
import json

app = Flask(__name__)

@app.route('/')
def index():
    with open('movie_title_index.json', 'r') as fp:
        movie_title_index = json.load(fp)
    movies=random.sample(list(movie_title_index.values()),3)
    return render_template('index.html', choices=movies)

@app.route('/recommendation')
def recommend():
    Rtrue=sql_data('input_data.csv')

    with open('movie_tags_index.json', 'r') as fp:
        movie_tags_index = json.load(fp)
    with open('movie_genres_index.json', 'r') as fp:
        movie_genres_index = json.load(fp)
    with open('movie_time_index.json', 'r') as fp:
        movie_time_index = json.load(fp)

    user_flask = dict(request.args)
    model_choice=dict(list(user_flask.items())[:1])
    model=list(model_choice.items())[0][1]
    pickle_file=list(model_choice.items())[0][1]
    time_block=dict(list(user_flask.items())[1:2])
    search_filter=list(time_block.items())[0][0]
    search_tag=list(time_block.items())[0][1]
    num_choice=dict(list(user_flask.items())[2:3])
    num=int(list(num_choice.items())[0][1])
    user_flask1=dict(list(user_flask.items())[3:])

    new_user_vector=user_input_vector(user_flask1, Rtrue)

    recommendations=recommender_build(model, Rtrue, pickle_file, new_user_vector, user_flask1, movie_time_index, movie_genres_index, movie_tags_index)

    movies=custom_recommendation(recommendations, search_filter, search_tag, num)

    return render_template('recommendation.html', movies=movies)


if __name__ == '__main__':
    #if I run "python application.py", please run the following code....
    app.run(debug=True, use_reloader=True)
