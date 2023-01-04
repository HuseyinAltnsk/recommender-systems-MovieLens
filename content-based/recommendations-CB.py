'''
Create content-based recommenders: Feature Encoding, TF-IDF/CosineSim
       using item/genre feature data


Team: Hüseyin Altınışık, Sebastian Charmot, Oğuzhan Çölkesen, and Eleni Tsitinidi.

Collaborator/Author: Carlos Seminario

sources:
https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.XoT9p257k1L

reference:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html


'''

import numpy as np
import pandas as pd
import math
import os
import pickle
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

SIG_THRESHOLD = 0 # accept all positive similarities > 0 for TF-IDF/ConsineSim Recommender
                  # others: TBD ...

def from_file_to_2D(path, genrefile, itemfile):
    ''' Load feature matrix from specified file
        Parameters:
        -- path: directory path to datafile and itemfile
        -- genrefile: delimited file that maps genre to genre index
        -- itemfile: delimited file that maps itemid to item name and genre

        Returns:
        -- movies: a dictionary containing movie titles (value) for a given movieID (key)
        -- genres: dictionary, key is genre, value is index into row of features array
        -- features: a 2D list of features by item, values are 1 and 0;
                     rows map to items and columns map to genre
                     returns as np.array()

    '''
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile:
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()

    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(movies), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}

    ##
    # Get movie genre from the genre file, place into genre dictionary indexed by itemID
    genres={}
    try:
        for line in open(path+"/"+genrefile, encoding="iso8859"):
            entry = line.split("|")
            genres[int(entry[1].rstrip())] = entry[0].rstrip()
    except Exception as ex:
        print (ex)
        print ("Proceeding with len(genres): ", len(genres))

    # Load data into a nested 2D list
    features = []
    start_feature_index = 5
    try:
        for line in open(path+'/'+ itemfile, encoding='iso8859'):
            #print(line, line.split('|')) #debug
            fields = line.split('|')[start_feature_index:]
            row = []
            for feature in fields:
                row.append(int(feature))
            features.append(row)
        features = np.array(features)
    except Exception as ex:
        print (ex)
        print ('Proceeding with len(features)', len(features))

    #return features matrix
    return movies, genres, features

def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file

        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name

        Returns:
        -- prefs: a nested dictionary containing item ratings (value) for each user (key)

    '''

    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile:
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()

    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(movies), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}

    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)

    #return a dictionary of preferences
    return prefs

def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})

            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def prefs_to_2D_list(prefs):
    '''
    Convert prefs dictionary into 2D list used as input for the MF class

    Parameters:
        prefs: user-item matrix as a dicitonary (dictionary)

    Returns:
        ui_matrix: (list) contains user-item matrix as a 2D list

    '''
    ui_matrix = []

    user_keys_list = list(prefs.keys())
    num_users = len(user_keys_list)

    itemPrefs = transformPrefs(prefs) # traspose the prefs u-i matrix
    item_keys_list = list(itemPrefs.keys())
    num_items = len(item_keys_list)


    sorted_list = True # <== set manually to test how this affects results

    if sorted_list == True:
        user_keys_list.sort()
        item_keys_list.sort()
        print ('\nsorted_list =', sorted_list)

    # initialize a 2D matrix as a list of zeroes with
    #     num users (height) and num items (width)

    for i in range(num_users):
        row = []
        for j in range(num_items):
            row.append(0.0)
        ui_matrix.append(row)

    # populate 2D list from prefs
    # Load data into a nested list

    for user in prefs:
        for item in prefs[user]:
            user_idx = user_keys_list.index(user)
            movieid_idx = item_keys_list.index(item)

            try:
                # make it a nested list
                ui_matrix[user_idx][movieid_idx] = prefs [user][item]
            except Exception as ex:
                print (ex)
                print (user_idx, movieid_idx)

    # return 2D user-item matrix
    return ui_matrix

def to_array(prefs):

    ''' convert prefs dictionary into 2D list
    -- prefs: dictionary containing user-item matrix'''
    R = prefs_to_2D_list(prefs)
    R = np.array(R)
    print ('to_array -- height: %d, width: %d' % (len(R), len(R[0]) ) )
    return R

def to_string(features):
    ''' convert features np.array into list of feature strings '''

    feature_str = []
    for i in range(len(features)):
        row = ''
        for j in range(len (features[0])):
            row += (str(features[i][j]))
        feature_str.append(row)
    print ('to_string -- height: %d, width: %d' % (len(feature_str), len(feature_str[0]) ) )
    return feature_str

def to_docs(features_str, genres):
    ''' convert feature strings to a list of doc strings for TFIDF '''

    feature_docs = []
    for doc_str in features_str:
        row = ''
        for i in range(len(doc_str)):
            if doc_str[i] == '1':
                row += (genres[i] + ' ') # map the indices to the actual genre string
        feature_docs.append(row.strip()) # and remove that pesky space at the end

    print ('to_docs -- height: %d, width: varies' % (len(feature_docs) ) )
    return feature_docs

def cosine_sim(docs):
    ''' Simple example of cosine sim calcs '''

    # tfidf invocation
    tfidf_vectorizer = TfidfVectorizer() # orig

    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)


    for i in range(tfidf_matrix.shape[0]):
        # get a vector
        vector_tfidfvectorizer=tfidf_matrix[i]
        # place tf-idf values in a pandas data frame
        df = pd.DataFrame(vector_tfidfvectorizer.T.todense(),
                          index=tfidf_vectorizer.get_feature_names(),
                          columns=["tfidf"])
        if i < 3: # print the first few of these to see what's happening
            print()
            print(df.sort_values(by=["tfidf"],ascending=False))
            print()
            print(tfidf_matrix[i])

    print()
    print('Document similarity matrix:')
    cosim_matrix = cosine_similarity(tfidf_matrix[0:], tfidf_matrix)
    print (type(cosim_matrix), len(cosim_matrix))
    # print('Examples of similarity angles')
    if tfidf_matrix.shape[0] > 2:
        for i in range(6):
            cos_sim = cosim_matrix[1][i] #(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))[0][i]
            if cos_sim > 1: cos_sim = 1 # math precision creating problems!
            angle_in_radians = math.acos(cos_sim)
    return cosim_matrix

def get_TFIDF_recommendations(prefs,cosim_matrix,user,movie_title_to_id,threshold):
    '''
            Calculates recommendations for a given user

            Parameters:
            -- prefs: dictionary containing user-item matrix
            -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
            -- user: string containing name of user requesting recommendation
            -- movie_title_to_id: dictionary that maps movie title to movieid
            --threshold: the similarity threshold that determines the neighborhood size of similarities
            Returns:
            -- rankings: A list of recommended items with 0 or more tuples,
               each tuple contains (predicted rating, item name).
               List is sorted, high to low, by predicted rating.
               An empty list is returned when no recommendations have been calc'd.

        '''
    notRatedMovies, recs = [], []   # array of tuples of not rated movies, array of recommendations
    #get non-rated movies
    for movie, movie_index in movie_title_to_id.items():
        if movie not in prefs[user].keys():
            notRatedMovies.append((movie, int(movie_index)-1))

    for movie,movie_index in notRatedMovies:
        sum_sim = 0
        sum_prod= 0
        for ratedMovie,rating in prefs[user].items():
            #get similarity from matrix
            sim = cosim_matrix[int(movie_title_to_id[ratedMovie])-1][movie_index]
            #if item in neighborhood to prediction calculation
            if sim>float(threshold):
                sum_sim+=sim
                sum_prod+= (sim*rating)
        if sum_sim!=0:
            recs.append((sum_prod/sum_sim,movie))

    return sorted(recs, key=lambda r: (r[0], r[1]), reverse=True) if recs != [] else recs

def get_FE_recommendations(prefs, features, movie_title_to_id, user):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- features: an np.array whose height is based on number of items
                     and width equals the number of unique features (e.g., genre)
        -- movie_title_to_id: dictionary that maps movie title to movieid
        -- user: string containing name of user requesting recommendation        
        Returns:
        -- rankings: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''

    notRatedMovies, recs = [], []   # array of tuples of not rated movies, array of recommendations
    for movie, movie_id in movie_title_to_id.items():
        if movie not in prefs[user].keys():
            notRatedMovies.append((movie, int(movie_id)))

    totalRatings = 0   # all ratings of user
    for movie, rating in prefs[user].items():
        movie_id = int(movie_title_to_id[movie])-1
        totalRatings += rating * np.count_nonzero(features[movie_id] == 1)

    # go over each non-rated movie to produce a prediction for it (if possible)
    for movie in notRatedMovies:

        movieGenreIndices = np.nonzero(features[movie[1]-1])[0]
        # create matrix with movie ratings for genres matching those of the non-rated movie
        # goes movie by movie and collects ratings on matching genres
        genreRatings = [[] for i in range(0, len(movieGenreIndices))] 
        for ratedMovie in prefs[user]:
            for i in range(0, len(movieGenreIndices)):
                movie_id = int(movie_title_to_id[ratedMovie])-1
                if features[movie_id][movieGenreIndices[i]] == 1:   # if rated movie has a matching genre
                    genreRatings[i].append(prefs[user][ratedMovie])
        
        # if there were no genre-sharing rated movies, no prediction can be made
        if genreRatings != [[] for i in range(0, len(movieGenreIndices))]:
            
            genreWeights = [np.sum(genreList)/totalRatings for genreList in genreRatings]
            totalWeight = np.sum(genreWeights)
            totalWeightContribs = [genreWeight/totalWeight for genreWeight in genreWeights]
            
            # multiply each genre's contribution with average of ratings on that genre
            genrePred = [np.average(genreRatings[i]) * totalWeightContribs[i] for i in range(0, len(genreRatings)) if totalWeightContribs[i] != 0]
            totalPred = np.sum(genrePred)
            recs.append((totalPred, movie[0]))

        else:
            print("No prediction can be generated for %s" % movie[0])  # since it has no matching genre ratings
    
    return sorted(recs, key=lambda r: (r[0], r[1]), reverse=True)[:10] if recs != [] else recs

def loo_cv_sim(prefs, cosim_matrix_or_features, algo, movie_title_to_id, threshold):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY

    Parameters:

    -- prefs dataset: critics, etc.
    -- cosim_matrix_or_features: : pre-computed cosine similarity matrix or features (aglorithm difference)
    -- sim: distance, pearson, etc.
    -- algo: the algorithm used - user-based recommender, item-based recommender, etc.
    --threshold: the similarity threshold that determines the neighborhood size of similarities

    Returns:
        error_total: 3 lists corresponding to MSE, MAE, or RMSE totals
	    error_list: 2 lists corresponding MSE+RMSE, MAE of actual-predicted differences
    """

    error_list_MSEorRMSE, error_list_MAE = [], []
    counter = 0

    for user, items in prefs.items():
        user_errors_MSEorRMSE, user_errors_MAE = [], []
        # show progress
        if (counter%1 == 0):
            print (counter, "/" , len(prefs))
        counter += 1

        for i in range(0, len(items)):
            # remove the item to be predicted
            loo_items = items.copy()
            popping = list(loo_items.keys())[i]
            actual = loo_items[popping]
            del loo_items[list(loo_items.keys())[i]]

            # execute algorithm with the loo dictionary
            prefs[user] = loo_items
            if algo == get_FE_recommendations:
                predicted_ratings = algo(prefs, cosim_matrix_or_features, movie_title_to_id, user)
            if algo == get_TFIDF_recommendations:
                predicted_ratings = algo(prefs, cosim_matrix_or_features, user, movie_title_to_id, threshold) # Change 0 for threshold
            try:
                prediction = [r[0] for r in predicted_ratings if r[1] == popping][0]
                if (prediction == 0):
                    print(r[1])

                user_errors_MSEorRMSE.append((prediction - actual)**2)
                user_errors_MAE.append(abs(prediction - actual))
            except:
                continue
            prefs[user] = items  # fixing the dictionary to have the removed item back

        error_list_MSEorRMSE = np.concatenate((error_list_MSEorRMSE, user_errors_MSEorRMSE))
        error_list_MAE = np.concatenate((error_list_MAE, user_errors_MAE))

        if len(user_errors_MSEorRMSE) > 0:
            error = np.average(user_errors_MSEorRMSE)
        else:
            error = -999

        # 'Avg' because this will print the same result for MSE and RMSE since they both use (prediction - actual)**2
        # print('Avg SE and SE results list for %s\t: %.3f' % (user, error),  list(map(lambda i: "%.3f" % i, user_errors_MSEorRMSE)))

        if len(user_errors_MAE) > 0:
            error = np.average(user_errors_MAE)
        else:
            error = -999

    print()
    total_MSE = np.average(error_list_MSEorRMSE)
    total_MAE = np.average(error_list_MAE)
    total_RMSE = math.sqrt(total_MSE)

    # error list or RMSE is the same as MSE's, so it's not added to the end of this return. Return 5 instead of 6
    return total_MSE, error_list_MSEorRMSE, total_MAE, error_list_MAE, total_RMSE   # 5 objects

def print_histogram(sim_matrix):
    """
    Prints the histogram of a given similarity matrix

    Parameters:
    -- sim_matrix: pre-computed similarity matrix


    """
    similarities = []
    for i in range(len(sim_matrix)):
        for j in range(i+1, len(sim_matrix[0])):
            similarities.append(sim_matrix[i][j])

    plt.hist(similarities, 10, facecolor='blue', range=(0,1), alpha=0.5)
    plt.xlabel('Similarity')
    plt.ylabel('# of occurances')
    plt.title('Histogram of similarities')
    plt.show()

def movie_to_ID(movies):
    ''' converts movies mapping from "id to title" to "title to id"
    Parameters:
    -- movies: a dictionary containing movies and their IDs '''
    return {v:k for k,v in movies.items()}


def main():

    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    print()
    prefs = {}
    features = []
    cosim_matrix = []
    done = False

    while not done:
        print()
        file_io = input('R(ead) critics data from file?, \n'
                        'RML(ead) ml100K data from file?, \n'
                        'FE(ature Encoding) Setup?, \n'
                        'TFIDF(and cosine sim Setup)?, \n'
                        'CBR-FE(content-based recommendation Feature Encoding),? \n'
                        'CBR-TF(content-based recommendation TF-IDF/CosineSim),? \n'
                        'LCV(leave one out cross validation),? \n'
                        'T(est of Hypothesis),? \n'
                        '==>> '
                        )

        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data' # for userids use 'critics_ratings_userIDs.data'
            itemfile = 'critics_movies.item'
            genrefile = 'critics_movies.genre' # movie genre file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            movie_title_to_id = movie_to_ID(movies)
            print('Number of users: %d\nList of users:' % len(prefs),
                  list(prefs.keys()))

            print ('Number of distinct genres: %d, number of feature profiles: %d' % (len(genres), len(features)))
            print('genres')
            print(genres)
            print('features')
            print(features)

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratngs file
            itemfile = 'u.item'  # movie titles file
            genrefile = 'u.genre' # movie genre file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            movie_title_to_id = movie_to_ID(movies)

            print('Number of users: %d\nList of users [0:10]:'
                  % len(prefs), list(prefs.keys())[0:10] )
            print ('Number of distinct genres: %d, number of feature profiles: %d'
                   % (len(genres), len(features)))
            print('genres')
            print(genres)
            print('features')
            print(features)

        elif file_io == 'FE' or file_io == 'fe':
            print()

            if len(prefs) > 0 and len(prefs) <= 10: # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)

                '''
                # e.g., critics data (CES)
                R = np.array([
                [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
                [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
                [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
                [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
                [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
                [0.0, 4.5, 0.0, 4.0, 1.0, 0.0],
                ])
                '''
                print('critics')
                print(R)
                print()
                print('features')
                print(features)

            elif len(prefs) > 10:
                print('ml-100k')
                # convert prefs dictionary into 2D list
                R = to_array(prefs)

            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == 'TFIDF' or file_io == 'tfidf':
            print()
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)
                feature_docs = to_docs(feature_str, genres)

                '''
                # e.g., critics data (CES)
                R = np.array([
                [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
                [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
                [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
                [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
                [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
                [0.0, 4.5, 0.0, 4.0, 1.0, 0.0],
                ])
                '''
                print('critics')
                print(R)
                print()
                print('features')
                print(features)
                print()
                print('feature docs')
                print(feature_docs)
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print('cosine sim matrix')
                print(cosim_matrix)

                '''
                <class 'numpy.ndarray'>

                [[1.         0.         0.35053494 0.         0.         0.61834884]
                [0.         1.         0.19989455 0.17522576 0.25156892 0.        ]
                [0.35053494 0.19989455 1.         0.         0.79459157 0.        ]
                [0.         0.17522576 0.         1.         0.         0.        ]
                [0.         0.25156892 0.79459157 0.         1.         0.        ]
                [0.61834884 0.         0.         0.         0.         1.        ]]
                '''

                #print and plot histogram of similarites
                # print_histogram(cosim_matrix)

            elif len(prefs) > 10:
                print('ml-100k')
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)
                feature_docs = to_docs(feature_str, genres)

                print(R[:3][:5])
                print()
                print('features')
                print(features[0:5])
                print()
                print('feature docs')
                print(feature_docs[0:5])
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print('cosine sim matrix')
                print (type(cosim_matrix), len(cosim_matrix))
                print()


                '''
                <class 'numpy.ndarray'> 1682

                [[1.         0.         0.         ... 0.         0.34941857 0.        ]
                 [0.         1.         0.53676706 ... 0.         0.         0.        ]
                 [0.         0.53676706 1.         ... 0.         0.         0.        ]
                 [0.18860189 0.38145435 0.         ... 0.24094937 0.5397592  0.45125862]
                 [0.         0.30700538 0.57195272 ... 0.19392295 0.         0.36318585]
                 [0.         0.         0.         ... 0.53394963 0.         1.        ]]
                '''

                #print and plot histogram of similarites)
                print_histogram(cosim_matrix)

            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == 'CBR-FE' or file_io == 'cbr-fe':
            print()

            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                print('critics')
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                recs = get_FE_recommendations(prefs, features, movie_title_to_id, userID)
                print()
                print("recs for %s:" % userID)
                print(recs)

            elif len(prefs) > 10:
                print('ml-100k')
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                recs = get_FE_recommendations(prefs, features, movie_title_to_id, userID)
                print()
                print("recs for %s:" % userID)
                for rec in recs: print(rec)

            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == 'CBR-TF' or file_io == 'cbr-tf':
            print()
            # determine the U-I matrix to use ..
            R = to_array(prefs)
            feature_str = to_string(features)
            feature_docs = to_docs(feature_str, genres)
            cosim_matrix = cosine_sim(feature_docs)
            threshold = input('Enter neighbor threshold')
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                print('critics')

                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                recs = get_TFIDF_recommendations(prefs,cosim_matrix,userID,movie_title_to_id,threshold)
                print()
                print("recs for %s:" % userID)
                for rec in recs: print(rec)


            elif len(prefs) > 10:
                print('ml-100k')
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                recs= get_TFIDF_recommendations(prefs,cosim_matrix,userID,movie_title_to_id,threshold)
                print()
                print("recs for %s:" % userID)
                for rec in recs: print(rec)

            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == "lcv" or file_io == "LCV":
            print()
            print('LOO_CV_SIM Evaluation')
            if len(prefs) == 7:
                prefs_name = 'critics'
            else:
                prefs_name = 'ML-100K'

            algo = input("Enter recommendation algorithm (TFIDF or FE): ")
            valid = True
            threshold = 0

            if (algo == "FE" or algo == "fe"):
                if (len(features) == 0):
                    valid = False
                    print("You need to run FE to generate features matrix first!")
                else:
                    total_MSE, list_MSE, total_MAE, list_MAE, total_RMSE = loo_cv_sim(prefs, features, get_FE_recommendations, movie_title_to_id, 0)
            elif (algo == "TFIDF" or algo == "tfidf"):
                if (len(cosim_matrix) == 0):
                    valid = False
                    print("You need to run TFIDF to generate cosine similarity matrix first!")
                threshold = input('Enter similarity neighbor threshold: ')
                total_MSE, list_MSE, total_MAE, list_MAE, total_RMSE = loo_cv_sim(prefs, cosim_matrix, get_TFIDF_recommendations, movie_title_to_id, threshold)

            else:
                valid = False
                print("Invalid algorithm entered. Try again.")

            if valid:
                pickle.dump(list_MSE, open( "error_list_%s_%s.p" % (algo.lower(), str(threshold)),  "wb" ))
                print('MSE for %s: %.5f, len(SE list): %d, using TFIDF'
                            % (prefs_name, total_MSE, len(list_MSE)) )
                print('MAE for %s: %.5f, len(SE list): %d, using TFIDF'
                            % (prefs_name, total_MAE, len(list_MAE)))
                print('RMSE for %s: %.5f, len(SE list): %d, using TFIDF'
                            % (prefs_name, total_RMSE, len(list_MSE)))
                print()

        elif file_io == "t" or file_io == "T":
            print()
            print('Test of Hypothesis')

            algo = input("Enter error_list1 algorithm (FE/TFIDF): ")
            threshold = 0
            if (algo == "tfidf" or algo == "TFIDF"):
                threshold = input("Enter error_list1 threhold: ")

            error_list1 = pickle.load(open( "error_list_%s_%s.p" % (algo.lower(), str(threshold)), "rb" ))

            algo = input("Enter error_list2 algorithm (FE/TFIDF): ")
            threshold = 0
            if (algo == "tfidf" or algo == "TFIDF"):
                threshold = input("Enter error_list2 threhold: ")

            error_list2 = pickle.load(open( "error_list_%s_%s.p" % (algo.lower(), str(threshold)), "rb" ))

            print()
            print ('t-test for error lists length => ',len(error_list1), " and ", len(error_list2))
            print ('Null Hypothesis is that the means (MSE values for User-LCV distance and pearson) are equal')

            ## Calc with the scipy function
            t_lcv, p_lcv = stats.ttest_ind(error_list1, error_list2)
            print("t = " + str(t_lcv))
            print("p = " + str(p_lcv), '==>> Unable to reject null hypothesis that the means are equal') # The two-tailed p-value

        else:
            done = True

    print('Goodbye!')

if __name__ == "__main__":
    main()
