import pandas as pd
from sklearn.decomposition import NMF
import numpy as np
import pickle
import json

def sql_data(input_csv):
    '''
    Takes the original data from postgres applies the specified filter and
    produces the necessary input data and required varibles for other functions.
    '''
    Rtrue=pd.read_csv(input_csv, index_col=0)
    num_movies=Rtrue.shape[1]
    movieId=Rtrue.columns
    return num_movies, movieId

def user_input_vector(user_flask, num_movies, movieId, movie_title_json):
    '''
    Take the the user input dictionary from flask and converts it into a
    proper array for recommender function.
    '''
    with open(movie_title_json, 'r') as fp:
        movie_title_index = json.load(fp)
    #create default vector
    new_user_vector = pd.DataFrame([2.5]*num_movies, index=movieId).reset_index()
    new_user_vector.columns=['movieId', 0]
    new_user_vector=new_user_vector.replace({'movieId':movie_title_index}).set_index('movieId').T
    #insert given user ratings to key movies
    for key, value in user_flask.items():
        new_user_vector.loc[:, key] = float(value)
    return new_user_vector, movie_title_index

def recommender_build(pickle_file, new_user_vector, user_flask, movie_time_index, movie_title_index, movie_genre_index, movie_tag_index, movieId):
    '''
    Builds a recommendation table based on model and user input array previously built.
    '''
    model = pickle.load(open(pickle_file, 'rb'))
    with open(movie_time_index, 'r') as fp:
        movie_time_index = json.load(fp)
    with open(movie_tag_index, 'r') as fp:
        movie_tag_index = json.load(fp)
    with open(movie_genre_index, 'r') as fp:
        movie_genre_index = json.load(fp)
    #calculate recommendations based on model components
    user_profile=model.transform(new_user_vector)
    results=np.dot(user_profile, model.components_)
    #customizing recommendation table with dictionary indices
    results2=pd.DataFrame(results[0]).set_index(movieId).reset_index()
    results2.columns=['movieId', 0]
    results2['genre']=results2['movieId'].map(movie_genre_index)
    results2['tag']=results2['movieId'].map(movie_tag_index)
    results2['time_block']=results2['movieId'].map(movie_time_index)
    results2.columns=['movieId', 'weight', 'genre', 'tag', 'time_block']
    recommendations=results2.replace({'movieId':movie_title_index})
    #remove already input movies
    for i in user_flask.keys():
            recommendations.drop(recommendations.index[recommendations['movieId'] == i], inplace = True)
    return recommendations

def custom_recommendation(recommendations, search_filter, search_tag, num=5):
    '''
    Customizes recommendation based on nothing, time_block or genre
    '''
    if search_filter == 'none':
        recs=np.array(recommendations.sort_values('weight', ascending=True)['movieId'][:num])
    if search_filter == 'time_block':
        recs=np.array(recommendations[recommendations['time_block']==search_tag].sort_values('weight', ascending=True)['movieId'][:num])
    if search_filter == 'genre':
        recs=np.array(recommendations[recommendations['genre']==search_tag].sort_values('weight', ascending=True)['movieId'][:num])
    return recs
