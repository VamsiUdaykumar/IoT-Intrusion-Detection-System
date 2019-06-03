
# coding: utf-8

# In[92]:


from sklearn.metrics import confusion_matrix
import pylab as pl
import numpy as np
import matplotlib.colors as mcolors
# Read the data
with open('expectedsig.txt', 'r') as infile:
    true_values = [int(i) for i in infile]
with open('predictedsig.txt', 'r') as infile:
    predictions = [int(i) for i in infile]
with open('expectedano.txt', 'r') as infile:
    true_values1 = [int(i) for i in infile]
with open('predictedano.txt', 'r') as infile:
    predictions1 = [int(i) for i in infile]

# Make confusion matrix
confusion = confusion_matrix(true_values, predictions)
confusion1 = confusion_matrix(true_values1, predictions1)
confusion2= confusion_matrix(predictions, predictions1)

print(confusion)
print(confusion1)
print(confusion2)


# In[93]:


classNames = ['Normal = 0','Attack = 1']
tick_marks = np.arange(len(classNames))
#confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
pl.matshow(confusion,cmap='Blues')
pl.title('Singature based IDS\n')
pl.colorbar()
pl.xlabel('\nPredicted')
pl.ylabel('Expected\n')
pl.xticks(tick_marks,classNames)
pl.yticks (tick_marks,classNames)
for i in range(2):
    for j in range(2):
        #confusion[i][j]=round(confusion[i][j],3)*100
        pl.text(j,i,str(confusion[i][j]))
pl.show()

#confusion1 = confusion1.astype('float') / confusion1.sum(axis=1)[:, np.newaxis]
pl.matshow(confusion1,cmap='Blues')
pl.title('Anomaly based IDS\n')
pl.colorbar()
pl.xlabel('\nPredicted')
pl.ylabel('Expected\n')
pl.xticks(tick_marks,classNames)
pl.yticks(tick_marks,classNames)
for i in range(2):
    for j in range(2):
        #confusion1[i][j]=round(confusion1[i][j],3)*100
        pl.text(j,i,str(confusion1[i][j]))
pl.show()

#confusion2 = confusion2.astype('float') / confusion2.sum(axis=1)[:, np.newaxis]
pl.matshow(confusion2,cmap='Blues')
pl.title('Predictions\n')
pl.colorbar()
pl.xlabel('\nby Anomaly')
pl.ylabel('by Singature\n')
pl.xticks(tick_marks,classNames)
pl.yticks(tick_marks,classNames)
for i in range(2):
    for j in range(2):
        #confusion2[i][j]=round(confusion2[i][j],3)*100
        pl.text(j,i,str(confusion2[i][j]))
pl.show()


