import random

MOVIES = ['Toy Story 1',
          'Toy Story 2',
          'Shawshank Redemption',
          'Toy Story 3',
          'SAW',
          'The Godfather',
          'Inception',
          'Some awesome Nicolas Cage Movie',
          'Python: The Movie',
          'Monty Python: Life of Brian']


def random_recommend(movies, num):
    random.shuffle(movies)
    return movies[:num]
