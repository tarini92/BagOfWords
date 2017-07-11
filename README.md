# BagOfWords
We use the Bag of words approach to do a Bike vs Horse Classification.
In computer vision, the bag- of-words model (BoW model) can be applied to image classification, by treating image features as words, where a bag of visual words is a vector of occurrence counts of a vocabulary of local image features. The following steps are employed in order to implement the bag-of-visual-words model:

1. Each image can be treated as a document, so there has to some represen- tation of words in the document, for which we perform - feature detection, feature extraction, and codebook generation.

2. The images are an abstraction of some local patches.These patches need to be represented as vectors, which is why we apply SIFT descriptor, to essentially turn each noticeable patch of an image to a 128-dimensional vector.

3. These patches are extracted and stored a vector, and they represent the features of the image.

4. The final step for the BoW model is to convert vector-represented patches to ”codewords” (analogous to words in text documents), which also pro- duces a ”codebook” (analogy to a word dictionary). A codeword can be considered as a representative of several similar patches. We perform k- means clustering over all the vectors. Codewords are then defined as the centers of the learned clusters. The number of the clusters is the codebook size (analogous to the size of the word dictionary).

5. Each patch in an image is mapped to a certain codeword through the clustering process and the image can be represented by the histogram of the codewords.

6. Now, to train the classifier, we first partition the available data into 85 and 25 percent training and testing data respectively. Then we train the BoW model using Logistic Regression algorithm.

7. Finally, we take the testing data,detect and compute its features using SIFT, and create a histogram of features, and pass it into the model for prediction.

8. The score of prediction done by Logistic Regression for the bikes and horses gives us 81 percent accuracy.
