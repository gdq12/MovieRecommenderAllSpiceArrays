import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import pickle

#import data
ratings=pd.read_csv('ml-latest-small/ratings.csv')
movies=pd.read_csv('ml-latest-small/movies.csv')
tags=pd.read_csv('ml-latest-small/tags.csv')

def sql_data(dfrate, dfmovie, dftag, filter_name):
    '''
    Takes the original data from postgres applies the specified filter and
    produces the necessary input data, dictionary indeces, number variables
    and array needed for other functions.
    '''
    dfrate1=dfrate.copy()
    dfmovie1=dfmovie.copy()
    dftags1=dftag.copy()

    #array for time_block dictionary index
    dfrate1['timestamp'] = pd.to_datetime(dfrate1['timestamp'], unit='s')
    dfrate1.set_index('timestamp', inplace=True)
    dfrate1.loc[dfrate1.between_time('00:00', '04:00').index, 'time_block']='cant sleep'
    dfrate1.loc[dfrate1.between_time('04:00', '07:00').index, 'time_block']='early commuter'
    dfrate1.loc[dfrate1.between_time('07:00', '11:00').index, 'time_block']='talk show'
    dfrate1.loc[dfrate1.between_time('11:00', '15:00').index, 'time_block']='kiddy nap'
    dfrate1.loc[dfrate1.between_time('15:00', '18:00').index, 'time_block']='afterschool'
    dfrate1.loc[dfrate1.between_time('18:00', '21:00').index, 'time_block']='tv dinner'
    dfrate1.loc[dfrate1.between_time('21:00', '23:59:59').index, 'time_block']='late night'
    dfrate1.reset_index(inplace=True)
    time_tag=pd.DataFrame(dfrate1.groupby('movieId')['time_block'].value_counts().unstack().idxmax(axis=1), columns=['time_block'])

    #filtering the input data
    #respective userid and movieId identifiers
    user_rate=pd.DataFrame(dfrate1.groupby('userId')['rating'].count())
    title_rate_count=pd.DataFrame(dfrate1.groupby('movieId')['rating'].count())
    title_rate_mean=pd.DataFrame(dfrate1.groupby('movieId')['rating'].mean())
    #filters
    bottom75_user=user_rate[user_rate['rating']<600].index
    top25_movies_voted=title_rate_count[title_rate_count['rating']>9].index
    #2
    if filter_name == 'bottom75_users_neutralize':
        dfrate1.set_index('userId', inplace=True)
        dfrate1.loc[bottom75_user,'rating']=3
        dfrate1.reset_index(inplace=True)
    #3
    if filter_name == 'top25_voted_neutralize':
        dfrate1.set_index('movieId', inplace=True)
        dfrate1.loc[top25_movies_voted, 'rating']=3
        dfrate1.reset_index(inplace=True)
    #5
    if filter_name == 'combo':
        dfrate1.set_index('userId', inplace=True)
        dfrate1.loc[bottom75_user,'rating']=3
        dfrate1.reset_index(inplace=True)
        dfrate1.set_index('movieId', inplace=True)
        dfrate1.loc[top25_movies_voted, 'rating']=3
        dfrate1.reset_index(inplace=True)

    #create input matrix
    Rtrue=dfrate1.pivot(index='userId', columns='movieId', values='rating').fillna(2.5)
    num_movies=Rtrue.shape[1]
    movieId=Rtrue.columns

    #creating index dictionaries
    movie_title_index=dict(zip(dfmovie1.movieId, dfmovie1.title))
    movie_genre_index=dict(zip(dfmovie1.movieId, dfmovie1.genres))
    movie_tag_index=dict(zip(dftags1.movieId, dftags1.tag ))
    movie_time_index=dict(zip(time_tag.index, time_tag.time_block))

    return movie_time_index, movie_title_index, movie_genre_index, movie_tag_index, num_movies, movieId

def user_input(user_flask, num_movies, movieId, movie_title_index):
    '''
    Take the the user input dictionary from flask and converts it into a
    proper array for recommender function.
    '''
    #create default vector
    new_user_vector = pd.DataFrame([2.5]*num_movies, index=movieId).reset_index()
    new_user_vector=new_user_vector.replace({'movieId':movie_title_index}).set_index('movieId').T
    #insert given user ratings to key movies
    for key, value in user_flask.items():
        new_user_vector.loc[:, key] = float(value)
    return new_user_vector

def recommender_build(model, new_user_vector, user_flask,movie_time_index, movie_title_index, movie_genre_index, movie_tag_index, movieId):
    '''
    Builds a recommendation table based on model and user input array previously built.
    '''
    #calculate recommendations based on model components
    user_profile=model.transform(new_user_vector)
    results=np.dot(user_profile, model.components_)
    #customizing recommendation table with dictionary indices
    results2=pd.DataFrame(results[0]).set_index(movieId).reset_index()
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


movie_time_index, movie_title_index, movie_genre_index, movie_tag_index, num_movies, movieId=sql_data(ratings, movies, tags, 'combo')

model = pickle.load(open('model5', 'rb'))

#Example user input
user_flask={'Grumpier Old Men (1995)': '5',
            'Jumanji (1995)': '3',
            'Waiting to Exhale (1995)': '4'}

new_user_vector=user_input(user_flask, num_movies, movieId, movie_title_index)

recommendations=recommender_build(model, new_user_vector, user_flask, movie_time_index, movie_title_index, movie_genre_index, movie_tag_index, movieId)

#example input
custom_recommendation(recommendations, 'time_block', 'cant sleep', 10)
