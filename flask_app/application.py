from flask import Flask, render_template, request
from recommender2 import sql_data, user_input_vector, recommender_build, custom_recommendation

app = Flask(__name__)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recommendation')
def recommend():
    num_movies, movieId=sql_data('combo_input.csv')

    user_flask = dict(request.args)

    new_user_vector, movie_time_index=user_input_vector(user_flask, num_movies, movieId, 'movie_title_index.json')

    recommendations=recommender_build(model, new_user_vector, user_flask, 'movie_time_index.json', movie_title_index, 'movie_genre_index.json', 'movie_tag_index.json', movieId)

    movies=custom_recommendation(recommendations, 'time_block', 'cant sleep')

    return render_template('recommendation.html', movies=movies)


if __name__ == '__main__':
    #if I run "python application.py", please run the following code....
    app.run(debug=True, use_reloader=True)
