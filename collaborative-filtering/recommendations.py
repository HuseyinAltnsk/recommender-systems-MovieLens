'''
Recommender System
Source info: PCI, TS, 2007, 978...

Author/Collaborator: Carlos Seminario

Researcher and Developer: Huseyin Altinisik, Sebastian Charmot, Oguzhan Colkesen, Eleni Tsitinidi

'''

import os, csv, numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pickle

def user_to_user_compute(prefs, func):
    ''' Calculate given User-User function for each distinct pair of users.
        Applies the given function to each pair.

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- func: function to be computed on the pair of users
      
        Returns: None
        
    '''

    # compare each distinct pair of users, based on the given function
    for i in range(0, len( list(prefs.keys()) )):
        for j in range(i+1, len( list(prefs.keys()) )):
            user1 = list(prefs.keys())[i]
            user2 = list(prefs.keys())[j]
            print('Computing %s for %s & %s: %f' 
                % (func.__name__, user1, user2, func(prefs, user1, user2)))

def create_item_set(prefs):
    '''
        Creates the item set to loop through

        Parameters:
        -- prefs: dictionary containing user-item matrix

        Returns:
        -- the item set
    '''

    item_set = set()

    for user in prefs:
        for movie in prefs[user]:
            item_set.add(movie)

    return item_set

def sim_pearson(prefs, person1, person2, significance_weighting=75): 
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        -- significance_weighting: significance weighting value (0 means no significance weighting)
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    '''

    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1

    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0

    # get the items rated by both users
    person1_subset = {item:rate for item,rate in prefs[person1].items() if item in prefs[person2]}
    person2_subset = {item:rate for item,rate in prefs[person2].items() if item in prefs[person1]}

    # get the average rating for each user
    person1_avg = np.average(list(person1_subset.values()))
    person2_avg = np.average(list(person2_subset.values()))

    # Add up the multiplication of all the differences using the mean values above, to calculate covariance
    covariance = sum([(person1_subset[item]-person1_avg)*((person2_subset[item]-person2_avg)) 
                        for item in person1_subset])
    # calculate product of std deviations of the two users, to get denominator
    denominator = sqrt( sum([pow((person1_subset[item]-person1_avg), 2) for item in person1_subset]) ) \
            *sqrt( sum([pow((person2_subset[item]-person2_avg), 2) for item in person2_subset]) )

    print("Significance weighting\t: %d" % significance_weighting)

    if denominator == 0:
        return 0
    else:
        if len(si) >= significance_weighting:
            return covariance/denominator
        else:
            return (len(si)/significance_weighting)*(covariance/denominator)

def sim_distance(prefs, person1, person2, significance_weighting=50):
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        -- significance_weighting: significance weighting value (0 means no significance weighting)
        
        Returns:
        -- Euclidean distance similarity as a float
        
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    # Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) 
                        for item in prefs[person1] if item in prefs[person2]])

    print("Significance weighting\t: %d" % significance_weighting)
    if (len(si) >= significance_weighting):
        return 1/(1+sqrt(sum_of_squares))
    else:
        return (len(si)/significance_weighting)*(1/(1+sqrt(sum_of_squares)))

def topMatches(prefs,person,similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        
        Returns:
        -- A list of similar matches with 0 or more tuples, 
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
        
    '''     
    scores=[(similarity(prefs,person,other),other) 
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def convert_to_rank_matrix(prefs):
    '''
        Uses the prefs matrix to replace the ratings given by users with how 
        that movie is ranked by that user with respect to other movies.

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A dictionary containing the matrix of ranks for all movies 
           rated by the users
    '''  

    rank_matrix = {}
    
    for user in prefs:
        temp_dict = {}
        ratings_list = []
        
        # Firstly, we create a list of all the times in a specific segment.
        for movie in prefs[user]:
            ratings_list.append(prefs[user][movie])
            temp_dict[movie] = 0
        
        # Then, we sort that list in ascending order (Faster lap time will be first).
        ratings_list.sort(reverse=True)
        rank_matrix[user] = temp_dict
        
        for movie in prefs[user]:
            # The index of a specific time is the rank of that time with respect to all times.
            # We add one to the rank because indexing starts from 0 in Python lists.
            indices = [i for i, x in enumerate(ratings_list) if x == prefs[user][movie]]
            index = np.average(indices)
            rank_matrix[user][movie] = index + 1
            
    return rank_matrix

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary) 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix, 
           this function returns an I-U matrix
        
    '''     
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print("%d / %d" % (c,len(itemPrefs)))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n)
        result[item]=scores
    return result

def calculateSimilarUsers(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of users showing which other users they are most
        similar to.

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)

        Returns:
        -- A dictionary with a similarity matrix

    '''
    result={}
    # Invert the preference matrix to be item-centric

    c=0
    for user in prefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0:
            print("%d / %d" % (c,len(prefs)))


        # Find the most similar items to this one
        scores=topMatches(prefs,user,similarity,n=n)
        result[user]=scores
    return result

def getRecommendedItems(prefs,itemMatch,user, threshold):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- itemMatch: item-item similarity matrix (nested dictionary)
        -- user: string containing name of user
        -- threshold: the threshold for the ratings that will be taken into account
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=threshold: continue            
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings

def getRecommendationsSim(prefs,userMatch,user,threshold,item_set):
    '''
        Calculates recommendations for a given user

        Parameters:
        -- prefs: dictionary containing user-user matrix
        -- userMatch: user-user similarity matrix (nested dictionary)
        -- user: string containing name of user
        -- threshold: the threshold for the ratings that will be taken into account
        -- item_set: the set of all movies in the dataset

        Returns:
        -- A list of recommended items with 0 or more tuples,
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
    '''

    scores={}
    totalSim={}
    # Loop over items
    for item in item_set:
      # Loop over items similar to this one
      if item not in prefs[user].keys():
          for (similarity,other) in userMatch[user]:
              if similarity>threshold and item in prefs[other]:
                  scores.setdefault(item,0)
                  scores[item]+=similarity*prefs[other][item]
                  totalSim.setdefault(item,0)
                  totalSim[item]+=similarity

    rankings=[(score/totalSim[item],item) for item,score in scores.items()]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings
           
def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    ''' 
    Print item-based CF recommendations for all users in dataset

    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)

    Returns: None
    
    '''
    
    for i in range(0, num_users):
        if i < len(prefs.keys()):
            user = list(prefs.keys())[i]
            print ('Item-based CF recs for %s, %s: ' % (user, sim_method.__name__), 
                            getRecommendedItems(prefs, itemsim, user,0)[0:top_N])

def loo_cv_sim(prefs, sim, algo, sim_matrix, item_set):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     
    Parameters:
    -- prefs dataset: critics, etc.
    -- sim: distance, pearson, etc.
    -- algo: user-based recommender, item-based recommender, etc.
    -- sim_matrix: pre-computed similarity matrix
    -- item_set: the set of all movies in the dataset

	 
    Returns:
        error_total: MSE, or MAE, or RMSE totals for this set of conditions
	    error_list: list of actual-predicted differences
    """

    error_list_MSEorRMSE, error_list_MAE = [], []
    
    thresh = 0.4
    print("Threshold\t: %f" % thresh)

    for user, items in prefs.items():
        user_errors_MSEorRMSE, user_errors_MAE = [], []

        for i in range(0, len(items)):
            # remove the item to be predicted
            loo_items = items.copy()
            popping = list(loo_items.keys())[i]
            actual = loo_items[popping]
            del loo_items[list(loo_items.keys())[i]]

            # execute algorithm with the loo dictionary
            prefs[user] = loo_items
            
            if algo == getRecommendationsSim:
                predicted_ratings = algo(prefs, sim_matrix, user, thresh, item_set) # Change 0 for threshold
            if algo == getRecommendedItems:
                predicted_ratings = algo(prefs, sim_matrix, user, thresh) # Change 0 for threshold

            try:
                prediction = [r[0] for r in predicted_ratings if r[1] == popping][0]

                user_errors_MSEorRMSE.append((prediction - actual)**2)
                # error_list_MSEorRMSE.append((prediction - actual)**2)     # Alternative to np.concatenate()
                user_errors_MAE.append(abs(prediction - actual))
            except:
                # print('(No prediction with positive similarity for user - movie: %s - %s)' % (user, popping) )
                continue
            prefs[user] = items  # fixing the dictionary to have the removed item back

        error_list_MSEorRMSE = np.concatenate((error_list_MSEorRMSE, user_errors_MSEorRMSE))
        error_list_MAE = np.concatenate((error_list_MAE, user_errors_MAE))

        if len(user_errors_MSEorRMSE) > 0:
            error = np.average(user_errors_MSEorRMSE)
        else:
            error = -999

        # 'Avg' because this will print the same result for MSE and RMSE since they both use (prediction - actual)**2
        # print('Avg SE and SE results list for %s\t: %.3f' % (user, error),  list(map(lambda i: "%.3f" % i, user_errors_MSEorRMSE)))
        if int(user) % 100 == 0:
            print("Processed user:\t%s" % user)
        if len(user_errors_MAE) > 0:
            error = np.average(user_errors_MAE)
        else:
            error = -999
        # print('MAE and AE results list for %s\t: %.3f' % (user, error),  list(map(lambda i: "%.3f" % i, user_errors_MAE)))
    
    print()
    total_MSE = np.average(error_list_MSEorRMSE)
    total_MAE = np.average(error_list_MAE)
    total_RMSE = sqrt(np.average(error_list_MSEorRMSE))

    # error list or RMSE is the same as MSE's, so it's not added to the end of this return. Return 5 instead of 6
    return total_MSE, error_list_MSEorRMSE, total_MAE, error_list_MAE, total_RMSE   # 5 objects
        
    # LOO_CV_SIM : (sim, metric) result
    # (sim_pearson, MSE) 1.42729    # (sim_pearson, MAE) 1.01258    # (sim_pearson, RMSE) 1.19469
    # (sim_distance, MSE) 1.06476   # (sim_distance, MAE) 0.76909   # (sim_pearson, RMSE) 1.03187

def get_all_UU_recs(prefs, threshold,item_set,sim=sim_pearson, num_users=10, top_N=5):
    ''' 
    Print user-based CF recommendations for all users in dataset

    Parameters
    -- prefs: nested dictionary containing a U-I matrix
    -- threshold: the threshold value for the ratings that will be taken into account
    -- item_set: the set of all movies in the dataset
    -- sim: similarity function to use (default = sim_pearson)
    -- num_users: max number of users to print (default = 10)
    -- top_N: max number of recommendations to print per user (default = 5)

    Returns: None
    '''

    userMatch = calculateSimilarUsers(prefs,top_N,similarity=sim)
    for user in prefs:
        print ('User-based CF recs for %s, %s: ' % (user, sim.__name__))
        print(getRecommendationsSim(prefs,userMatch,user,threshold,item_set))

def loo_cv(prefs, metric, similarity, algo):
    """
        Leave_One_Out Evaluation: evaluates recommender system ACCURACY
        
        Parameters:
        -- prefs dataset: critics, ml-100K, etc.
        -- metric: MSE, MAE, RMSE, etc.
        -- similarity: distance, pearson, etc.
        -- algo: user-based recommender, item-based recommender, etc.
        
        Returns:
        -- error_total: MSE, MAE, RMSE totals for this set of conditions
        -- error_list: list of actual-predicted differences
    """
    error_list = []
    for user, items in prefs.items():
        user_errors = []
        for i in range(0, len(items)):

            # remove the item to be predicted
            loo_items = items.copy()
            popping = list(loo_items.keys())[i]
            actual = loo_items[popping]
            del loo_items[list(loo_items.keys())[i]]

            # execute algorithm with the loo dictionary
            prefs[user] = loo_items
            predicted_ratings = algo(prefs, user, similarity)

            try:
                prediction = [r[0] for r in predicted_ratings if r[1] == popping][0]
                err = (prediction - actual)**2
                user_errors.append(err)
                error_list.append(err)
            except:
                print('(No prediction with positive similarity for %s-%s)' % (user, popping) )
                continue
            prefs[user] = items  # fixing the dictionary to put the removed item back
            
        if metric.upper() == 'MSE':
            if len(user_errors) > 0:
                error = np.average(user_errors)
                print('MSE and SE results list for %s\t: %.3f' % (user, error),  list(map(lambda i: "%.3f" % i, user_errors)))
            else:
                error = -999

    error = np.average(error_list)
    return error, error_list

def data_stats(prefs, item_set):
    ''' Calculates and prints key statistics about the users and items, as well as
        computing a histogram of ratings

        Parameters:
        -- prefs: a nested dictionary containing item ratings for each user
        -- item_set: the set of all movies in the dataset
        
        Returns: None

    '''
    ratings, movie_dict, user_dict = [], {}, {}

    for user in prefs:
        # add every user's average rating
        user_dict[user] = np.average(list(map(lambda i: float(i), prefs[user].values())))

        for movie in prefs[user]:
            # collect all ratings for each movie
            movie_rating = float(prefs[user][movie])
            if movie not in movie_dict.keys():
                movie_dict[movie] = [movie_rating]
            else:
                movie_dict[movie].append(movie_rating)
            ratings.append(movie_rating)

    # basic statistics (all users and items)
    num_ratings, max_rating = len(ratings), max(ratings)
    avg_rating, std_dev = np.average(ratings), np.array(ratings).std()
    # average item rating and item std deviation (all items)
    item_ratings = np.array( [np.average(movie_dict[movie]) for movie in movie_dict] )
    avg_item_rating, item_std_dev = np.average(item_ratings), item_ratings.std()
    # average user rating and user std deviation (all users)
    user_ratings = list(user_dict.values())
    avg_user_rating, user_std_dev = np.average(user_ratings), np.array(user_ratings).std()

    # item count and matrix sparsity
    item_count = len(item_set)
    matrix_sparsity = 100*(1-(num_ratings/(len(prefs)*item_count)))
    
    # print all statistics
    print('Number of users: %d\nNumber of items: %d\nNumber of ratings: %d' % (len(prefs), item_count, num_ratings))
    print('Overall average rating: %.2f out of %d, and std dev of %.2f' % (avg_rating, max_rating, std_dev))
    print('Average item rating: %.2f out of %d, and std dev of %.2f' % (avg_item_rating, max_rating, item_std_dev))
    print('Average user rating: %.2f out of %d, and std dev of %.2f' % (avg_user_rating, max_rating, user_std_dev))
    print('User-Item Matrix Sparsity: %.2f%%' % matrix_sparsity)

    # compute the histogram
    plt.hist(ratings, 4, facecolor='g')
    plt.xlabel('Rating')
    plt.ylabel('Number of user ratings')
    plt.title('Rating Histogram')
    plt.xlim(0.5, 5.5)
    plt.ylim(0, 70000)
    plt.show()

def popular_items(prefs):
    ''' Calculates and prints the items based on popularity by most number of ratings,
        highest ratings and overall best ratings

        Parameters:
        -- prefs: a nested dictionary containing item ratings for each user

        Returns: None

    '''

    item_dict = {}
    for user in prefs:
        for movie in prefs[user]:
            if movie not in item_dict.keys():
                item_dict[movie] = [prefs[user][movie]]
            else:
                item_dict[movie].append(prefs[user][movie])
    
    i = 1
    # print most rated items
    print('\nPopular items -- most rated:')
    print('Title\t\t\t#Ratings\tAvg Rating')
    most_rated = {k:v for k,v in sorted(item_dict.items(),
                    key=lambda item: len(item[1]), reverse=True)}
    for item, ratings in most_rated.items():
        print('%s\t%d\t\t%.2f' % (item, len(ratings), np.average(ratings)))
        i = 1 if i == 5 else i+1
        if i == 1:
            break

    # print highest rated items
    print('\nPopular items -- highest rated:')
    print('Title\t\t\tAvg Rating\t#Ratings')
    highest_ratings = {k:v for k,v in sorted(item_dict.items(), 
                        key=lambda item: np.average(item[1]), reverse=True)}
    for item, ratings in highest_ratings.items():
        print('%s\t%.2f\t\t%d' % (item, np.average(ratings), len(ratings)))
        i = 1 if i == 5 else i+1
        if i == 1: 
            break
        
    # print overall best rated items
    print('\nOverall best rated items (number of ratings >= 5):')
    print('Title\t\t\tAvg Rating\t#Ratings')
    num_ratings = {k:v for k,v in sorted(item_dict.items(),
                    key=lambda item: np.average(item[1]), reverse=True) if len(v)>=5}
    for item, ratings in num_ratings.items():
        print('%s\t%.2f\t\t%d' % (item, np.average(ratings), len(ratings)))
        i = 1 if i == 5 else i+1
        if i == 1: 
            break

def printMatrix(itemsim):
    '''
        Prints the item similarity matrix

        Parameters:
        -- itemsim: the item similarity matrix

        Returns: None
    '''

    # initial space for first row
    tabRow = len(max(itemsim.keys()))
    tabRowSpace = ''.join(' ' * tabRow)
    # print matrix
    print(tabRowSpace + '|'.join(itemsim.keys()))

    for item1 in itemsim.keys():
        line = item1 + ''.join(' ' * (tabRow-len(item1)))
        for item2 in itemsim.keys():
            # sp(ace) is used to determine how much space to print before and after value of each cell
            if item1 == item2:
                sp = ''.join(' ' * ((len(item2)-1)//2))
                line += sp + '1' + sp
            else:
                try:
                    num = '{0:.3f}'.format([i[0] for i in itemsim[item1] if i[1]==item2][0])
                except:
                    num = '-'
                sp = ''.join(' ' * ((len(item2)-len(str(num)))//2) )
                line += sp + num + sp
            line += '|'
        print(line)
        
def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding="ISO-8859-1") as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
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
                
def main():
    ''' User interface for Python console '''
    
    # Load critics dict from file
    path = os.path.dirname(os.getcwd()) # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs, usersim, itemsim, rank_matrix = {}, {}, {}, {}

    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, '
                        'P(rint) the U-I matrix?, '
                        'S(tats)?, \n'
                        'D(istance) critics data?, '
                        'PC(earson Correlation) critics data?, '
                        'U(ser-based CF Recommendations)?, \n'
                        'Sim(ilarity matrix) calc for Item-based recommender?, '
                        'I(tem-based CF Recommendations)?, '
                        'LCVSIM(eave one out cross-validation)?, \n'
                        'SimU(ilarity matrix) calc for User-based recommender?'
                        'C(onvert) to rank matrix for Spearman?')
                        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'midterm/data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            item_set = create_item_set(prefs)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = '/midterm/movies/ml-100k/'
            datafile = 'u.data'
            itemfile = 'u.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            item_set = create_item_set(prefs)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))

        elif file_io == 'S' or file_io == 's':
            # Print general count and popularity stats
            print()
            if len(prefs) > 0:
                print('Calculating statistics from "%s" file' % datafile)
                data_stats(prefs, item_set)
                popular_items(prefs)
            
        elif file_io == 'D' or file_io == 'd':
            # Calc U-U Euclidian similarities
            print()
            if len(prefs) > 0:
                print('User-User distance similarities, for all users:')
                user_to_user_compute(prefs, sim_distance)
        
        elif file_io == 'PC' or file_io == 'pc':
            # Calc U-U Pearson correlations
            print()
            if len(prefs) > 0:             
                print('User-User Pearson correlation coefficients, for all users:')
                user_to_user_compute(prefs, sim_pearson)

        elif file_io == 'U' or file_io == 'u':
            # Calc User-based CF recommendations for all users                
            print()
            if len(prefs) > 0:             
                print('User-based CF recommendations for all users:')
                get_all_UU_recs(prefs, 0, item_set, sim_distance) ### hard-code threshold

        elif file_io == 'C' or file_io == 'c':
            # Calc User-based CF recommendations for all users                
            print()
            if len(prefs) > 0:             
                rank_matrix = convert_to_rank_matrix(prefs)
                print(rank_matrix)
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print() 
            if len(prefs) > 0:
                print ('Printing "%s" dictionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or RS(ead) spearman or WD(rite) distance or WP(rite) pearson or WS(rite) spearman? ')
                # itemsim = {}
                
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = sim_distance
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = sim_pearson
                    
                    elif sub_cmd == 'RS' or sub_cmd == 'rs':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_spearman.p", "rb" ))  
                        sim_method = sim_pearson
                        
                    elif sub_cmd == 'WS' or sub_cmd == 'ws':
                        if (len(rank_matrix) > 0):
                            # transpose the U-I matrix and calc item-item similarities matrix
                            itemsim = calculateSimilarItems(rank_matrix,similarity=sim_pearson)                     
                            # Dump/save dictionary to a pickle file
                            pickle.dump(itemsim, open( "save_itemsim_spearman.p", "wb" ))
                            sim_method = sim_pearson
                        else:
                            print("C(onvert) to rank_matrix first!")
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = sim_distance
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" )) 
                        sim_method = sim_pearson

                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter S(im) again and choose a Write command')
                    print()

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method.__name__, len(itemsim)))
                    # printMatrix(itemsim)
                    print()
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'SIMU' or file_io == 'simu':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or RS(ead) spearman or WD(rite) distance or WP(rite) pearson or WS(rite) spearman? ')
                # itemsim = {}
                
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_distance.p", "rb" ))
                        sim_method = sim_distance
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_pearson.p", "rb" ))  
                        sim_method = sim_pearson

                    elif sub_cmd == 'RS' or sub_cmd == 'rs':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_spearman.p", "rb" ))  
                        sim_method = sim_pearson
                        
                    elif sub_cmd == 'WS' or sub_cmd == 'ws':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        if len(rank_matrix) > 0:
                            usersim = calculateSimilarUsers(rank_matrix,similarity=sim_pearson)                     
                            # Dump/save dictionary to a pickle file
                            pickle.dump(usersim, open( "save_usersim_spearman.p", "wb" ))
                            sim_method = sim_pearson
                        else:
                            print("C(onvert) to rank_matrix first!")
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_distance.p", "wb" ))
                        sim_method = sim_distance
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_pearson.p", "wb" )) 
                        sim_method = sim_pearson
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter S(im) again and choose a Write command')
                    print()

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method.__name__, len(itemsim)))
                    # printMatrix(itemsim)
                    print()
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'I' or file_io == 'i':
            print()
            if len(prefs) > 0 and len(itemsim) > 0:                
                # Calc Item-based CF recommendations for all users
                print('Item-based CF recommendations for all users:')
                get_all_II_recs(prefs, itemsim, sim_method)
                print()
            else:
                if len(prefs) == 0:
                    print ('Empty dictionary, R(ead) in some data!')
                else:
                    print ('Empty similarity matrix, use Sim(ilarity) to create a sim matrix!')    

        elif file_io == "LCVSIM" or file_io == "lcvsim":
            # Calc leave-one-out cross validation
            print()
            if len(prefs) > 0:             
                print('LOO_CV_SIM Evaluation')
                if len(prefs) == 7:
                    prefs_name = 'critics'
                else:
                    prefs_name = 'full dataset'
                
                done2 = False
                while not done2:
                    algo = input('Enter U(ser) or I(tem) algo: ') # algo choice?
                    if algo.upper() == 'U':
                        algo = getRecommendationsSim
                        if usersim != {}:
                            similarity_matrix = usersim
                        else:
                            print ('Empty Sim Matrix, run SimU!')
                            break
                    elif algo.upper() == 'I':
                        algo = getRecommendedItems
                        if itemsim != {}:
                            similarity_matrix = itemsim
                        else:
                            print ('Empty Sim Matrix, run Sim!')
                            break
                    else:
                        print ("Invalid input! Try again.")
                        continue

                    done2 = True

                    total_MSE, list_MSE, total_MAE, list_MAE, total_RMSE = loo_cv_sim(prefs, sim_method, algo, similarity_matrix, item_set)
                    print('MSE for %s: %.5f, len(SE list): %d, using %s' 
                                % (prefs_name, total_MSE, len(list_MSE), sim_method.__name__) )
                    print('MAE for %s: %.5f, len(SE list): %d, using %s' 
                                % (prefs_name, total_MAE, len(list_MAE), sim_method.__name__) )
                    print('RMSE for %s: %.5f, len(SE list): %d, using %s' 
                                % (prefs_name, total_RMSE, len(list_MSE), sim_method.__name__) )
                    print()

                    if prefs_name == 'critics':
                        print('MSE/RMSE error list:', list_MSE)
                        print('MAE error list:', list_MAE)
            else:
                print ('Empty dictionary, run R(ead)!')
        else:
            done = True
    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()