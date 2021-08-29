
According to Arthur Samuel (1959),
> It is a field of study that gives computer ability to learn without being explicitly programmed.

But, how is a computer supposed to learn? Tom Mitchell says that
>A computer program is said to learn from experience E with respect to task T and some performance measure P, if its performance on T, as measured by P improves with experience E.

For example: Playing Checkers (Btw Arthur Samuel's Checkers playing program was among the world's first successful self-learning programs.) 

```
Here,
E = the experience of playing many games of checkers
T = the task of playing checkers
P = the probability that the program will win the next game.
```

Broadly, there are three types of Machine Learning Algorithms:
1. Supervised Learning
1. Unsupervised Learning
1. Reinforcement Learning

## Supervised Learning:

Here, we are the teachers. We tell the system what the correct answer is and train it until the system begins to predict with an acceptable level of accuracy. We can further group it into Regression and Classification.

**Regression**: Suppose, you have a dataset of students' reading habits and their performance in the exam. Then you plot 'Hours Studied' vs 'Test Grade'.
![Hours Studied vs Test Grade](/images/what-is-machine-learning/hour_vs_test.png)

You can clearly see that if you studied more you can get good grades. So, the system devises a function Y = f(x) [or ŷ = wx + b where w is the slope, x is hours studied and b is y-intercept]. We train the system to devise a function with minimum error possible. Then, if a new student comes and claims that he/she studies 5 hours a day, we can predict that he can get a certain test grade.

**Classification**: Let's suppose we have a dataset for breast cancer and we have features like Age and Tumor Size.
![Classification Supervised](/images/what-is-machine-learning/classification_supervised.png)

Based on our data, the algorithm will create a mapping function Y = f(x). Then, when we provide new data, it will predict whether breast cancer is malignant or benign based on Tumor Size and Age. Generally, we will have many more features (Clump thickness, Uniformity of Cell Size, Uniformity of Cell Shape, etc) to determine whether 1(Malignant) or 0(Benign) [Binary Classification].

## Unsupervised Learning: 

Here, we are not given the right answer. Here's the data. Do what you want to do with it. Algorithms are left on their own to devise and discover the interesting structure in the data. We can further group it into Clustering and Association Algorithms.
![Clustering](/images/what-is-machine-learning/clustering.png)

1. **Clustering Algorithm**: It groups similar things together. We don't provide any labels but the system understands the data and can differentiate it based on features it finds. One example where clustering is used is Google News. It groups similar news from different sites in cohesive groups.

2. **Association Algorithm**: Here is the data, find an association.
![Association](/images/what-is-machine-learning/association.jpeg)

    At a basic level, it analyzes data for patterns, or co-occurrence in a database and identifies if-then associations which are called association rules. In a grocery store, if we know that certain items are frequently bought together and they are on the same shelf, buyers of one item would be prompted to buy another. Promotional discounts and advertisements could then be forced on the customer's throat. 

    Let's consider, itemset1 = {bread, milk} & itemset2 = {bread, shampoo}. You can guess that itemset1 will have higher support (a measure of how frequent itemset is in all transactions) than itemset2. 
    If a cart has {bread} in it, {milk} has higher confidence (a measure of likeliness of occurrence of itemset if cart already has items) than {shampoo}.

## Reinforcement Learning: 

Here, the machine is exposed to an environment where it trains itself continually using trial and error. This machine learns from past experiences and tries to capture the best possible knowledge to make accurate business decisions.
Eg: You have an agent and reward, with many hurdles in between.
![Robot,Diamond,Fire](/images/what-is-machine-learning/robot_diamond_fire.png)
The goal of the robot here is to get to the reward (diamond) avoiding the hurdles (fire). The robot learns by trying all the possible paths and choosing the path which gives him the reward with the least hurdles. Each right step will give the robot a reward and each wrong step will subtract the reward of the robot. The total reward will be calculated when it reaches the final reward.