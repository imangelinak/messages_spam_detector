# SPAM Messages Detector Using Logistic Regression and Naive Bayes Model
Hello Angelina here,
This past week I've been working on a project that is detecting spam messages through big data. I've used the Logistic regression to detect the difference in the messages. The messages are named as "spam" for the spam messages and "ham" for the unidentified, not spam or missing data messages. 
I will explain the project as a whole in details, just so I can visualize the project better. 
This project I created using VScode, using Python 3.13.2 on a Jupyter Notebook for visualizing the data graphs. 
In the beginning, I downloaded and imported the data set as a csv file, into the Jupyter Notebook. 
This particular data set I got from kaggle.com. I find Kaggle helpful for a lot of projects for beginners as it allows to not only have a data set, but a person can find helpful information for the project that they are working on. 
The second step is to import all the libraries that are going to be used during the project: 
#### 1. Pandas
#### 2. Numpy 
#### 3. Seaborn
#### 4. Matplotlib
#### 5. Spacy
When it comes to Spacy I did have some difficulties importing the library into my project, as it was always giving me an error. However, after quiet a research I installed spacy and then imported it, which fixed my bug. 
After fixing the problem with Spacy, I had to read and project my data set table. 
Since the data that I downloaded had many columns, categories and data that I actually did not need for this projec I just had to use the V1 and V2 columns which respectfully I defined as "label" and "message", removing the rest of the unnecessary information. 
After, decluttering the data and saving what I needed, I wanted to check what is the differencial number between the "ham" and "spam" messages. Which one would be more in the data set? 
Using a bar chart I was able to visualize the difference between the count between both variables. The Bar Chart has the name "Distribution of Spam vs. Ham Messages" and on the Y-axis has the count, while on the X-axis has the two message types "spam" and "ham". 
From the visualized data through Bar Chart, we can see that the "ham" messages were dominating with 4825 vs the "spam" messages that are 747. 
After, defining which messages were more, I continued to clean the data to make it easily readable and destinguishable. 
I used TfidVectorizer to transofrm the data form a raw text document into a numerical representation that the machine learning algorithm that I will use later will understand and read. 
After that I imported the Logistic Regression Algorithm and started training it accordingly. 
I created a Confusion Matrix to predict the accuracy of the algorithm. 
From the Confusion Matrix we can see that the algorithm is giving a very high average score of accuracy 97%, as well as high average F1 - score (the performance) 96%. Considering the fact that there is a class imbalance in the csv file. 
I used GridSearchCV to make sure I'm on the right track and the Logistc Regression is the right algorithm to use. GridSearchCV showed that the best Logistic Regression model will be C=100 and with balanced class weight, considering the fact the data is not as balanced as it should be. 
After configuring the right algorithm and model, I created a ROC curve, which is a diagram that is representing the performance between the True Positive Rate (TPR) and the False Positive Rate (FPT). 
I did the ROC curve to represent how well the model can distinguish between the different classes. I also included treshhold to minimize the difference between the two. 
In order to have an accurate curve, I had to calculate the Area under the Curve also known as AUC. The specification of AUC is that the higher it is the more accurate the model is in distinguishing between the classes. After doing the AUC, I started plotting the ROC curve in order to get to the visualization. The diagram gave me an AUC of 0.99 which considered a great score. 
Another thing that I put on the ROC curve is a Random classifier just so I can have a base for comparison and provide a better visualization of the ROC curve and the results. 
The next thing that I did is to create another type of visualization that is going to give me predictions about the "spam" and "ham" words that are used in our data set. 
The visualization is a horizontal bar chart, and the colors that are indicating the "spam" and "ham" words are respecfully red for "spam" and green for "ham". This is the best way to visualize it, becuse psychologically "spam" is something bad which is represented by red and "ham" is represented as words that are indicating a non-spam messages and it is giving the "green light" in the project. 
Not only the colors are indicating the difference between "good" and "bad", I also made the colors gradient, indicating which specific word is more likely to be used in a "spam" or "ham" text. 
From the horizontal bar chart we can see that the most likely the word "txt" is a number one word used in "spam" message. Contrary, the word "my" is considered a word that would be used in "ham" message. 
This helped me categorize the words that are used in the messages, and helped finding the "parasite" words to categorize which message is "spam" or "ham". 
The next step that I did is using the Naive Bayes Model. 
After finding out which words are a red flag, the last thing I was supposed to do is to find where exatly those words are in the text. This is why I decided to use Naive Bayes Model, which is helping to find a data point's place in the data set. 
After coding the model, I had to check how accurate it is and if it is performing well. It turned out the model is almost 88% accurate which is considered a well score overall, but can be causing some bugs depending on the data. The problem is probably coming from some of the data not being processed.
After finding out how well is our model performing, I extracted a sample with few of the text messages that are detecting the "spam" words from the horizontal bar graph before.

# Conclusion 
