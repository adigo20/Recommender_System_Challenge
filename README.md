# Recommender_System_Challenge OverView of the steps performed. 
Recommender system using Different Techniques and comparision

Task -1 Recommender System Challenge
The challenge in task 1 was to build a recommender system to help us recommend using the user-item interaction data. The data set was collected from the social media platform
Flickr. Over aim is to recommend the items to the user with the help of the dataset given to us.

1.1 Understanding the dataset
• Training data - Contains interactions between users and items (photos). If a user liked a photo.
• Test data - Each user is provided with a list with 100 candidate photos in the test dataset, you will need to check the candidate list and recommend the top 15 photos for each user.
• Validation data. A validation dataset to be used to tune the model.
• (reference – Taken from the Kaggle competition description)

# Number of unique User are 3466 and Number of unique items are 9004 in total.

1.2 Processing the data for analysis

While looking at the train data which has user data , Items and rating. The rating is flagged as 1 or 0. The train data however only has ratings flagged as 1 which means there isn’t any data for 0. This basically means we have only one type of data. The validation has both 0 and 1 data which provides us a better data to train our model. This will also help us in reducing the bias in the train data. Hence the train and validation data was merged and made as new training data. There where few duplicates, this was handled by removing them. The final data count was 454846 after merging the data. After carefully making the dataset we then made the sparse matrix which can be then used for our algorithms. 

1.3 Algorithms
We had many different algorithms that could be used to build our recommendation system. The broad classification is content based and collaborative filtering methods. These two are the type of approaches that we could use for our analysis. We also have Explicit and Implicit feedback approach which are based on the metric to be evaluated in our data amoung User and Item. For the data which we have been provided in particular we can see it is a interaction-based data. Which is between the users and the item. Hence, it would be ideal to use Implicit feedback approach.

There are broadly 3 main implicit feedback based methods that we can use which are –
Alternating Least Squares, Logistic Matrix Factorisation and Bayesian personalised ranking
that can be used to evaluate the current data and recommend items to users.

Alternating Least Squares (ALS ) –
ALS is an iterative optimisation process. What is means is for every iterations we try to
arrive closer to the factorised representation of the original data.
The initial step involves making an interaction matrix. In our case we have sparse data so we
will make a sparse matrix . This sparse user-item matrix will be used for recommendations.
We then build the model and fit the data and then get recommendation using the in-built
function.
Initial Evaluation –
On implementing the ALS algorithm initially with all the parameters set at default We
achieved a score of 0.15 approx. The parameters used were -
• Alpha –increases the dimension of the interactions for better prediction (the model
sets default to 32)
• Factor –latent factors to compute (100)
• regularization – regularisation (0.01)
• iterations – no of iterations (15)
The basic model was somewhat not very accurate with respect to the recommendations.

Fine Tuning -
To increase our accuracy we are using precision score to analyse the models. To do so We used implicit evaluation precision_at_k function. 
This gives us the precision of our model.

Top 5 parameter combinations are should in the code file.
![image](https://user-images.githubusercontent.com/94940044/143151427-bcc818f9-7d39-44d6-9eee-62fe16261f21.png)



We can see that the model3 parameter combination works the best. We then used it on our final model.Doing so the best score which we got for ALS model 0.19729. Although there could be many more possible combinations of parameters the model 3 gave the best results.

Logistic Matrix Factorization (LMF) –
It is a collaborative filtering recommender model that learns probabilistic distribution. This means it learns by the probability distribution whether user will like it or not in particular. In our case where we have user data with interaction with the items this algorithm sounds promising.
Similar to ALS the initial step involves making an interaction matrix. In our case we have sparse data so we will make a sparse matrix. This sparse user-item matrix will be used for recommendations. We then build the model and fit the data and then get recommendation using the in-built function. On implementing the LM algorithm initially with all the parameters set at default We achieved a score of 0.16 approx.

The default key parameters for the LMF algorithm are
• Factor –latent factors to compute (30)
• Learning rate – learning rate to be applied for updates during training (0.85)
• regularization – regularisation (2.0)
• iterations – no of iterations (30)

Fine tuning –
We used implicit.evaluation ‘s precision_at_k function. This gives us the precision of our model.
Top 5 parameter combinations are should in the code file.

![image](https://user-images.githubusercontent.com/94940044/143151526-6105f6ed-e482-4dee-9ba9-f739cbb3ee39.png)



The model-4 had the precision score.
On using the parameters and uploading the results on Kaggle. The score which was achieved
was 0.21128.
LMF model gave the best score with respect to the Kaggle score.
Bayesian personalised ranking (BPR) –
The BPR algorithm incorporates Bayesian analysis using the likelihood function.
It considers each user and describes it by a feature.
Each item is described as a feature. For eg in our case it might say how close is the item to
the picture liked by the user. This is a where good approach if you are trying to rank a set of
people. This rank can widely be used where you want to rank people or customers in
relation to their liking or their peers liking.
The BPR model initially gave score of 0.13.
The default key parameters for the BPR algorithm are
• Factor –latent factors to compute (100)
• Learning rate – learning rate to be applied to updates during training (0.01)
• regularization – regularisation (0.01)
• iterations – no of iterations (100)
Fine tuning –
Even after fine tuning the parameters the BPR model didn’t give any better score than 0.13. 

The reason could be in the way the BPR model works, which is rank based. For our data set and the problem BPR isn’t the best way to recommend. The model could work better if we
had used additional data that was given to us. Although it is not certain that it will still out perform ALS and LMF.

Conclusion –
Both ALS and LMF models are best suited models for the task. The approach that LMF model Uses that is learns probabilistic distribution whether the user will like it or not gives it a little upper hand over ALS, hence giving us the best score of 0.21. Although, there are many other Deep-learning algorithms that can be used and could be a better way to be used for recommendation.


Task -2 Node Clustering
For the task 2 node clustering task we were asked to cluster the nodes in the network into several categories, and evaluate the performance of different clustering algorithms. At least one embedding approach was to be used.

2.1 Understanding the dataset
We were provided with 3 different files that we were to use for the clustering task.
Docs.txt – The file contains the title for each and every node in the network. Each node
represents a paper.
Labels.txt – The file contains class labels for all the node to be used to know the type of
node for the normalized mutual info score.
Adjedges.txt – the file contains the adjacent node information to which the node is
connected.
Algorithms –
Node2Vec –
The node2vec algorithm generates vectors of the nodes on the graphs. The framework
learns this representation using random walk through the graph starting at the target
variable. The whole aim is to preserve the neighbouring node and to represent the graph as real vector. The neighbouring node has small Euclidean distance as well as is well represented.

1) K-means with Node2Vec –
The k-means algorithm takes the data points as input and them makes a group into kclusters. The most alike datapoints are put in same cluster. The grouping process is done in
the training phase.
The basic principle is that we add a new point to the data called centroids. The centroid will
try to stay in the middle of 1 k cluster. The moving stops when algorithm stops.
We were asked to find the Normalized Mutual Information Score (NMIS).
The NMIS score is a measure that evaluates the network portioning performed by the
algorithm. The higher the NMIS the better. The range is from 0 to 1 with 1 being the most
accurate. The NMIS is measure of strength association in two nodes.
The NMIS for k-means was 0.44 with the basic parameters. Which a pretty good score.


2) Agglomerative Clustering with Node2Vec –
It is a hierarchal based clustering that tries to merge every single cluster into a one whole cluster. It assigns each data point as a single cluster. Distance matrix is used to find the distance, this process repeats up until one big cluster is formed. The NMIS was 0.44.
Both K-means and agglomerative Clustering are opposite in the approach of clustering. Kmean we use a centroid to walk through the datapoints and make clusters while in case of
Agglomerative clustering starts from all points as single node and reducing the count of
nodes to 1.We will further use K-mean as it gave a little better NMI score. 

Spectral with K-means –
The spectral clustering uses the eigenvalues of similarity matrix for dimensionality reduction before performing clustering. Spectral clustering allows non-graphical data to be clustered unlike K-means. No assumptions are made while forming the clusters. The NMI score is very low.

Text embedding –
In the text embedding we first converted all the text to lower and then removed the special characters , extra spaces and digits. Doing this we had only the text data. We then remove the stopwords and lemmatize the data. Post lemmatizing we create vectors and fit the data on K-means algo using fit predict(). The NMI score is 0.11.

Conclusion -
The Node2Vec with K-means performed the best among all the algorithm because it considers global view of the network as it considers depth first search and also breadth first
search.


Reference –
• https://www.mygreatlearning.com/blog/introduction-to-spectral-clustering/
• https://implicit.readthedocs.io/en/latest/lmf.html
• https://scikitlearn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.
html
• https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation
• https://realpython.com/k-means-clustering-python/
