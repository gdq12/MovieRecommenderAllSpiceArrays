import pandas as pd
from sklearn.decomposition import NMF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def sql_data(input_csv):
    '''
    Imports premade input historical user data applicable to all models.
    '''
    Rtrue=pd.read_csv(input_csv, index_col=0)
    return Rtrue

def user_input_vector(user_flask, Rtrue):
    '''
    Take the the user input dictionary from flask and converts it into a
    proper array for recommender function.
    '''
    new_user_Id=len(Rtrue)+1
    new_user_vector = pd.DataFrame([2.5]*Rtrue.shape[1], index=Rtrue.columns).T
    for key, value in user_flask.items():
        new_user_vector.loc[:, key] = float(value)
    new_user_vector.rename(index={0:new_user_Id},inplace=True)
    return new_user_vector

def recommender_build(model, Rtrue, pickle_file, new_user_vector, user_flask, movie_time_index, movie_genres_index, movie_tags_index):
    '''
    Builds a recommendation table based on model and user input array previously built.
    '''
    if model == 'cosim_model':
        new_user_Id=len(Rtrue)+1
        R_new_user=Rtrue.append(new_user_vector)
        simi_user_matrix=pd.DataFrame(cosine_similarity(R_new_user), index=R_new_user.index, columns=R_new_user.index)
        simi_user_vector=simi_user_matrix[new_user_Id][~(simi_user_matrix.index==new_user_Id)]
        results2=pd.DataFrame(np.dot(simi_user_vector, Rtrue)/simi_user_vector.sum(), index=Rtrue.columns).reset_index()
    if model == 'bottom75_user_neutral' or model == 'top25_movies' or model == 'combo':
        model1 = pickle.load(open(pickle_file, 'rb'))
        user_profile=model1.transform(new_user_vector)
        results=np.dot(user_profile, model1.components_)
        results2=pd.DataFrame(results[0]).set_index(Rtrue.columns).reset_index()
    results2.columns=['movieId', 'weight']
    results2['genre']=results2['movieId'].map(movie_genres_index)
    results2['tag']=results2['movieId'].map(movie_tags_index)
    results2['time_block']=results2['movieId'].map(movie_time_index)
    recommendations=results2
    for i in user_flask.keys():
        recommendations.drop(recommendations.index[recommendations.movieId== i], inplace = True)
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
