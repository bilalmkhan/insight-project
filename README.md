# Project for Insight Data Science Fellowship

## SurveySense: Measuring Workplace Satisfaction from a Large Employee Text Survey

Background: My client is a company in the senior care assisted-living industry employing over 100,000 people. There is research that shows that worker satisfaction is a strong predictor of satisfaction among residents in the senior care industry. Therefore, the company wants to measure employee job satisfaction. In addition, they also want to know what aspects of their jobs the employees are happy or unhappy about.

The company carried out a large survey of employees asking them two open ended questions: What aspects of their jobs are they happy about and what needs improvement. As a result, they ended up with over 200,000+ open-ended text responses.
Since it is impossible for the company’s management to go through all these responses, I was recruited to help them make sense of the large amount of data.

The company is particularly interested in finding out employee satisfaction in some key areas such as employee benefits, employee salary, support of coworkers, employee relations, organized workplace, work schedules, staffing levels, etc.

Here is the link to the final dashboard that I delivered: https://tinyurl.com/surveysense

As can be seen from the dashboard, all comments are categorized in the 11 topics of interest identified by the company. We can see for example that employees are happy about their supportive coworkers and unhappy about their pay and feel the workplace is understaffed. There are also 6 randomly selected examples of comments classified in each topic area.

The main challenge in carrying out this project was the lack of any labelled data. Since many comments fell in more than one category, an unsupervised learning approach such as clustering was not feasible. Moreover, since there are 11 topic areas, I needed a large labelled dataset to train a supervised model with good results.

To overcome this challenge I came up with a way to label the data. While going through the comments, I noticed that certain phrases could precisely predict their respective topic class. For example, phrases such as, ‘health insurance’, ‘PTO’, and ‘401K’ were always used in the context of Employee Benefits. On the other hand, the word benefits itself was not a precise predictor of the topic class employee benefits.cI wrote a python script that found such phrases and assigned the comments their respective topic labels. This gave me a labeled dataset of over 30,000 comments that I could use to carry out text classification using supervised machine learning.

For text classification, I used a deep learning model with ELMO embeddings. ELMO assigns vectors to words so that they can be used by a machine learning algorithm. ELMO embeddings, which were developed be the Allen Institute for Artificial Intelligence in Seattle, are deep contextualized word embeddings. This means that they are able to assign different vectors to identical words based on their surrounding context. For example, ELMO embeddings can distinguish between the two different uses of the word benefits in these two sentences. I chose ELMO over other deep contextualized word embeddings such as BERT because ELMO is trained at the character level whereas BERT is trained at the subword level. This allows ELMO to assign the correct vectors even to mistyped words which were very common in my data. 

The employee comments are fed to the ELMO embedding which converts them into contextualized word vectors and passes them on to a feed-forward network which outputs the probability of the comment belonging to different classes. 

To evaluate the model, I used the F1-score because the F1-score takes into account precision and recall which were both important in our case. The trained topics classification model had a macro-average F1 score above 90% on the test set. For sentiment classification, I trained a binary classifier with an F1-score above 95% on the test set. 

