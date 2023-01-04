__For the code to run, move the data/ folder and the 6 error list files in the same directory as the Python file.__

# Experiment Results - CB
If the variations of content-based recommendation algorithms in terms of both accuracy and coverage were to be ranked, the order would be:
1. FE = TF-IDF, >0 = TF-IDF, >0.25
2. TF-IDF, >0.50
3. TF-IDF, >0.75
4. TF-IDF, =1

### Concluding Remarks
This result did not fully align with the initial hypothesis that TF-IDF will perform better than FE due to the inverse document frequency term. TF-IDF is not worse than FE either, they are just evenly matched in our experiment in both accuracy and coverage metrics. Moreover, the team realized the computation of the FE algorithm was faster than TF-IDF by a couple of factors in our implementations. __Given that this is the only advantage one algorithm has over the other, we claim that FE is the best content-based algorithm choice for our given dataset.__

__The contradiction with our initial hypothesis shows that our assumption that inverse document frequency will make TF-IDF perform better than FE is incorrect.__ Even though the inverse document frequency is not explicitly included in the FE algorithm, the influence of each movie over the user preference vectors is lower for genres that the user has seen more of in the past. Therefore, both TF-IDF and FE basically implemented similar concepts in our experimental method. More importantly, note that we generated the text files for each movie by putting all the genres of the movie in a space-separated string (e.g. "Comedy Drama Romance"). In this implementation, each genre only occurred once in the item strings that the TF-IDF algorithm works on. In other words, term frequency was always 1 for each genre of a movie, simply because we could not have an item string such as "Action Action". Consequently, the TF-IDF vectors were only dependent on IDF, because term frequency did not matter (it was only 0 or 1). If we had a news article, we would have different term frequencies for each word (e.g. "the": 47, "we": 21, etc.), so the term frequencies for each item would influence the item vectors in diversified ways. __In general, this analysis suggests that term frequency is the defining quality of TF-IDF over FE. This is why TF-IDF is better suited for naturally written text items, such as news articles and books, while FE is better suited for movies due to its fast computation and clearly-defined features that only occur once.__

Additionally, we also expected that the similarity threshold would not result in better recommendations, which our results agreed with. The main reason for this is as we increase the similarity threshold, we include fewer data points to use in our recommendations. Having a smaller subset of data available to generate recommendations results in both lower coverage and accuracy. Our results and explanation align with previous work as well. [1]

Furthermore, related to the experiment with 6 different recommender systems, one of our hypotheses was that matrix factorization methods would produce the most meaningful results, because of its 100% coverage and high accuracy as a pure recommendation method. Our results confirmed this hypothesis and allowed us to distinguish between ALS and SGD. Firstly, matrix factorization recommenders were able to satisfy the coverage and serendipity criteria better than other methods. __In addition to that, SGD produced more varied predictions while predictions of ALS were consistently at the lower end of the range if not the lowest. Therefore, although inconclusive, our intuition is that SGD is a more reliable MF method than ALS.__

- - - -

1. _Jonathan L. Herlocker, Joseph A. Konstan, Al Borchers, and John Riedl. 1999. An Algorithmic Framework for Performing Collaborative Filtering. In
Proceedings of the 1999 Conference on Research and Development in Information Retrieval. 230–237._
