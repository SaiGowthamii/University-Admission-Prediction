
Abstract
In today’s competitive educational landscape, accurately predicting
the chances of getting admission into universities has
become paramount. The aim of this study is to develop a
predictive model that assesses the likelihood of admission
based on scores of GRE, Toefl, Cgpa etc..Chosen data-set
contains continuous variables, we opted to implement various
machine learning algorithms, such as Linear Regression,
Decision Trees, Random Forest Regression, and Neural Network,
in order to analyze and compare their performance.
This analysis aims to determine the best-performing model
and shed light on the ambition or safety of the selected university.
1. Introduction
Preparing for the Graduate Program is a complex task that
requires careful planning and decision-making. Many students
who apply to master’s programs struggle to select suitable
universities, either due to a lack of awareness about
university evaluation criteria on bases of giving admission
to the students or mislead of information gathered from different
source. In the process of university admissions, understanding
the evaluation criteria employed by educational
institutions is of utmost importance for aspiring students.
In this study, we propose a Machine Learning approach to
predict university admissions by training and testing on a
diverse range of student profiles. By leveraging the power
of data analysis and predictive modeling, we aim to shed
light on the factors that significantly influence admission
outcomes.
Our approach utilizes a dataset sourced from Kaggle, consisting
of 8 columns and 401 rows. Each column represents
a distinct attribute or feature, providing valuable information
about the applicants, while each row represents a unique
instance or sample within the dataset. By leveraging this
dataset, we aim to uncover patterns and relationships that
can aid in predicting admission outcomes accurately.
To ensure robust evaluation and accurate predictions, we
divided the data-set into training and testing data using the
K-Fold algorithm, with five folds. This approach allows us
to train our model on a subset of the data and validate its
Copyright © 2023, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
performance on unseen samples, thus mitigating the risk of
over-fitting and enhancing generalization capabilities.
By employing this Machine Learning methodology, we
aim to uncover patterns and relationships within the data-set
that can provide valuable insights into the admission process.
Ultimately, our goal is to develop a predictive model
that can assist prospective students in gauging their chances
of admission to their desired universities.
In the subsequent sections, we will delve into the details
of our methodology, feature analysis, model selection, and
the results of our predictive model. Through this study, we
hope to contribute to the field of university admissions by
leveraging the power of Machine Learning to enhance the
transparency and efficiency of the decision-making process.
2. Why did we choose the problem? Why is it
important to us?
We chose this problem based on our personal experience
and the challenges we faced during the university application
process. Like many other students, we went through
a phase of uncertainty and anxiety when applying to universities.
We often found ourselves wondering about our
chances of admission and investing significant amounts of
money in applications without a clear understanding of our
performance.The lack of transparency in the evaluation process
left us feeling frustrated and helpless. We believe that
no student should have to go through the same experience.
Therefore, it is important to us to address this problem and
provide a solution that can alleviate the stress and uncertainty
surrounding university admissions.
3. Data Exploration
Our dataset[1] consists of 401 instances, each representing a
unique student profile, sourced from Kaggle. The dataset encompasses
eight columns, with each column representing a
distinct attribute or feature relevant to the admission process.
These attributes could include GRE Score, TOEFL Score,
University Rating, SOP, LOR , CGPA , Research , Chance of
Admit typically consider during their decision-making process.
Feature description :
GRE Score:
An integer value that shows the score of GRE exam which
in range of 10-360.
TOEFL Score:
An integer value that shows the score of TOEFL exam which
in range of 10-120.
University Rating:
A float value that describes the university rating in range of
0-5.0
SOP:
A float value that represents the rating of Statement of Purpose
in range of 0-5.0
LOR:
A float value that represents the rating of letter of recommendation
in range of 0-5.0
CGPA:
A float value that represents the cumulative grade points
which is in range of 0.0-10.0.
Research:
A Boolean value which represent 1 as in research and 1 as
in.
Figure 1: Heatmap showing the correlation among the features
By Observing the heat map CGPA and GRE Score has
the highest positive correlation with the chance of admit
which means the applicants with higher CGPA and GRE
score were most likely to be accepted to the universities.
Given Below Figures represents how each feature consider
in relation to Chance of Admit:
Figure 2: Feature label vs Target label
By Observing from the plots above , we can analyze that
Gre score above 310, Toefl Score above 105, Cgpa above 8.5
and Students did Research have 0.8, University with rank
4, Sop with 4.0 points, Lor with 4.0 points have above 0.8
Chance of getting admit.
4. Experimental Analysis
The dataset has been analyzed using regression models,
including Linear Regression, Decision Tree Regression,
Neural Network, and Gradient Boosting. We used K-Fold
validation on dataset.The actual and predicted values are
compared to determine errors, which are then compared to
both scikit-learn algorithms and a reference paper[2]. The
goal is to identify the best algorithm from among these
methods.
Since the dependent variable is continuous data, accuracy,
precision, recall, and F1 score metrics used in
classification problems are not suitable. Moreover, determining
a threshold for admission decisions based on various
factors and the complex relationship between predictor
variables and the dependent variable makes developing an
accurate model difficult.
a. Linear Regression
As the data has continuous numerical variables, and the
dependent variable is also continuous. It assumes a linear
relationship between the independent and dependent variables.
So, we choose linear regression.
Table - 1 : Comparing with various learning rate
Alpha MSE MSE-std deviation
0.01 0.03183711 0.00152943
0.1 0.00754285 0.00221700
0.05 0.01543734 0.00178138
Table - 2: Evaluation Metrix
MSE - MSE - Skl LR-MSE [2]
0.00754285 0.00417477 0.00480149
In Table 1, the Mean Squared Errors for each learning rate
are shown.Additionally, the findings are taken from the
linked publication and the Scikit-Learn algorithm, and they
are compared with our best learning rate result in table 2.
Figure 3: Importance Vs Feature
The most significant element in the graph above is CGPA.
When compared to other features, this variable will be given
a lot of weight.
Figure 4: Actual label Vs Predicted label - Our algorithm
Figure 5: Actual label Vs Predicted label - Using scikit learn
As a result, we get to the conclusion that our MSE is
0.003 MSE higher than that of Scikit Learning and the
findings of published studies. Our results weren’t as strong
as those in the original study since we didn’t incorporate
any notion of learning rate degradation.
b. Decision Tree Regression
Decision tree regression can handle numerical features such
as GRE Score, TOEFL Score, CGPA, and also categorical
features such as University Rating, SOP, LOR, and Research
experience. It can split the dataset based on these features
and their values to create a tree-like structure that can
predict the target variable, which is the Chance of Admit.
We performed hyperparameter tuning for the decision tree
regression using K-fold validation. The results are presented
in the table below.
Table - 3 : Comparing with max-depth
Max-Depth MSE-avg SD Skl-MSE
3 0.00557 0.000931 0.009920
4 0.01251 0.001423 0.019315
5 0.01922 0.001458 0.027765
6 0.02665 0.001515 0.036554
7 0.03541 0.001751 0.044681
From above Table-1 , we can observe that, Max-depth of
3 has less mean square error compare to the other MSE’s
in both our own implementation and Sklearn.Now in below
Table-2 will compare the MSE of max-depth 3 with SKlearn
and paper[2].
Table - 4 : Evaluation Matrix
MSE MSE - Skl DTR-MSE[2]
0.00557 0.000931 0.00874299
By Observing Table-2, Our own implementation has less
mean square error compared with MSE-Paper[2]. But has
more mean square error compared with SKlearn.
Figure 6: MSE-Error VS Max-Depth
Here, we can observe that , increasing of the max-depth
it is taking more number of decisions where it is leading
to over-fitting. Instead of learning it memorizes as increase
the depth. Testing error increase gradually as max-depth increases.
Generated Decision trees [3] for each fold of each maxdepth
3,4,5,6,7.
c.Random Forest Regression
Random Forest not only delivers precise predictions but
also grants valuable insights regarding the significance of
various features in relation to the outcome variable, thereby
providing a measure of feature importance.In this algorithm
we mainly focused on feature improtance of our data-set.
We did this implementation using Sklearn.
Figure 7: Feature Importance
We can Observe the feature importance, CGPA has highest
feature importance according to our data-set compare with
other features and lowest feature importance to Research.
Table - 5 : Evaluation Matrix
MSE - Skl RFR-MSE[2]
0.0041167 0.00582112
We can Observe that mean square error of ours is bit less
than the MSE-Paper[2].
e. Gradient Boosting
Gradient Boosting Regression is a suitable algorithm for this
data because it can handle both numerical and categorical
data, and can handle non-linear relationships between the
predictor variables and the target variable. Additionally,
Gradient Boosting Regression can handle missing data
and outliers in the dataset, which is important for ensuring
accurate predictions.
Since the target variable, ”Chance of Admit”, is a continuous
variable, the use of regression algorithms like Gradient
Boosting Regression is appropriate for predicting the target
variable for a single point.
Table - 7 : Comparing with various loss functions
Loss function used MSE
sq-error 0.00312296
abs-error 0.00296461
huber 0.00300180
identity 0.00543008
Gradient boosting is applied with different loss functions,
including sq-error, abs-error, huber, and identity, using
Scikit. When we use abs as the loss function after comparing,
the loss is minimal. Additionally, cross-validation was
used to add all of the value to the data.
Figure 8: Actual label Vs Predicted label - Using scikit learn
Figure 9: Actual label Vs Predicted label - Using scikit learn
We can Observe the feature importance from the above
graph, CGPA has highest feature importance according to
our data-set compare with other features and lowest feature
importance to Research. But we got difference in other
features rank. some got increased and some decreased in
their importance.
e. Neural Network
The use of activation functions in neural networks allows
them to model non-linear relationships between input and
output variables, which can make them more effective at
solving complex problems than linear models. Activation
functions are applied to the output of each neuron to introduce
non-linearity We have used four activation functions.
We tried with different layers with different number of
perceptrons. And the best results are obtained when we used
1-layered network with 2 perceptrons each.
Figure 10: Actual label Vs Predicted label - Using scikit
learn
Table - 8 : Comparing with various activation functions
Activation Function MSE
relu 0.02249441
identity 0.64279623
tanh 0.679297527
logistic 0.700115507
Here, we can observe the average mean squared error for
different activation functions and concluded that the ”relu”
activation neural network has less MSE compared with all
other because it is simple, non-linear, efficient, and effective
for many types of neural network architectures and tasks.
5. Comparisons with various models
In this section we are comparing all the implemented
models by us.Find below table with all the algorithms with
their MSE and type of implementation.
Table - 9 : Comparing all algorithms with MSE
Implement Type Algorithms MSE
Own Linear Regression 0.00754285
Own Decision Tree Regression 0.00557
Sklearn Linear Regression 0.00417
Sklearn Decision Tree Regression 0.000931
Sklearn Random Forest Regression 0.00411
Sklearn Gradient Boosting 0.00296
Sklearn Neural Network 0.02249441
6. Observations and Discussions
In conclusion, we applied K-fold validation to the dataset,
and the provided algorithms produced different mean
squared error (MSE) values, which reflect their respective
prediction accuracy.Compared all the algorithms implemented
by us and discussed the observation below
Figure 11: MSE VS Algorithms
• The results show that the ”Sklearn Decision Tree Regression”
algorithm achieved the lowest MSE of 0.000931,
suggesting it performed exceptionally well in accurately
predicting the target variable. This can be attributed to the
decision tree’s ability to capture complex relationships in
the data, making it a powerful algorithm for regression
tasks.
• The implemented ”Decision Tree Regression” algorithm
also performed reasonably well with an MSE of 0.00557.
While it had a higher MSE compared to the Sklearn version,
it still demonstrated effective predictive capabilities.
However, it is important to note that the implementation
details and parameters used in our algorithms could have
influenced their performance.
• The ”Sklearn Gradient Boosting regression” algorithm
achieved an MSE of 0.00296, indicating its strong performance
in reducing prediction errors. Gradient boosting
combines multiple weak models (typically decision
trees) to create a strong ensemble model, which likely
contributed to its superior accuracy.
• The ”Sklearn Random Forest Regression” and ”Sklearn
Linear Regression” algorithms had MSE values of
0.00411 and 0.00417, respectively. These values suggest
that both algorithms performed well in predicting the
target variable but were slightly less accurate compared
to the decision tree and gradient boosting models. Random
forests leverage the power of multiple decision trees,
while linear regression focuses on linear relationships between
features and the target variable.
• The implemented ”Linear Regression” algorithm had an
MSE of 0.00754285, indicating slightly higher prediction
errors compared to the Sklearn version. It is possible that
differences in implementation, feature selection, or regularization
techniques contributed to this variance in performance.
And our implementation lack the degradation
of learning rate, this could be reason for high error comparitively.
• Finally, the ”Sklearn Neural Network” algorithm demonstrated
the highest MSE of 0.1119. Neural networks are
powerful models capable of learning complex patterns,
but they require careful tuning of architecture, hyperparameters,
and training procedures. The high MSE suggests
that either the neural network was not appropriately
optimized for the given task or the dataset might not be
well-suited for neural network-based regression.
Overall, the MSE values provide insights into the performance
of each algorithm, highlighting their relative
accuracy in predicting the target variable. Selecting the
most appropriate algorithm that is decision tree regression
that has the MSE of 0.000931.
7. References
1. Kaggle Graduate Admissions Dataset.
2. M. S. Acharya, A. Armaan and A. S. Antony, ”A Comparison
of Regression Models for Prediction of Graduate
Admissions,” 2019 International Conference on Computational
Intelligence in Data Science (ICCIDS), Chennai,
India, 2019, pp. 1-5, doi: 10.1109/ICCIDS.2019.8862140. .
3. Generated Decision trees of K-folds for each depth.
