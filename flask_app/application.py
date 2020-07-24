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
    num_movies, movieId=sql_data('combo_input.csv')

    user_flask = dict(request.args)
    model_choice=dict(list(user_flask.items())[:1])
    time_block=dict(list(user_flask.items())[1:2])
    num_choice=dict(list(user_flask.items())[2:3])
    user_flask1=dict(list(user_flask.items())[3:])

    new_user_vector, movie_title_index=user_input_vector(user_flask1, num_movies, movieId, 'movie_title_index.json')

    recommendations=recommender_build(list(model_choice.items())[0][1], new_user_vector, user_flask1, 'movie_time_index.json', movie_title_index, 'movie_genre_index.json', 'movie_tag_index.json', movieId)

    movies=custom_recommendation(recommendations, list(time_block.items())[0][0], list(time_block.items())[0][1], int(list(num_choice.items())[0][1]))

    return render_template('recommendation.html', movies=movies)


if __name__ == '__main__':
    #if I run "python application.py", please run the following code....
    app.run(debug=True, use_reloader=True)
