## Description:

In this logs folder, I put three logs to record the results for 3 slightly different kinds of dataset.

sentivent_5e5.err: based on the original paper, the amount of docs is Train : Dev : Test = 228 : 30 : 30.

sentivent_shuffles.err: The amount of docs is Train : Dev : Test = 228 : 30 : 30, but I shuffles the docs to re-split the dataset in order to solve the unbalance of labels. 

sentivent_shuffles2.err: In order to solve that the model cannot be trained fully, I change the amount of docs is Train : Dev : Test = 258 : 15 : 15.

