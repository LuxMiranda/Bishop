import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('white')

color_Bayesian = '#C0504D'
color_Analogy = '#4F81BD'
color_Human = '#9BBB59'
color_NUC = '#E87217'


t = [1,2,3]
analogy = [0.77, 0.81, 0.96]
nuc = [0.83, 0.82, 0.95]
bayesian = [0.7, 0.89, 0.91]
human = [0.77, 0.89, 1.00]

data = pd.DataFrame({
        'Timestep' : t,
        'Analogy' : analogy,
        'NUC' : nuc,
        'Bayesian' : bayesian,
        'Human' : human
        })

print(data)

plt.grid(True)
g = sns.lineplot(x=data['Timestep'], y=data['Human'], color=color_Human,linewidth=3,marker='^',markersize=10,label='Human')
g = sns.lineplot(x=data['Timestep'], y=data['Analogy'], color=color_Analogy,linewidth=3,marker='D', markersize=8,label='Analogy')
g = sns.lineplot(x=data['Timestep'], y=data['Bayesian'], color=color_Bayesian,linewidth=3,marker='s',markersize=8,label='Bayesian')
g = sns.lineplot(x=data['Timestep'], y=data['NUC'], color=color_NUC,linewidth=3,marker='*',markersize=14, label='NUC')
g.set(xlim = (0.75, 3.25))
g.set_xticklabels(['','1','','2','','3'])
g.set(ylim = (0.65, 1.04))
plt.ylabel('Accuracy')
plt.legend()
plt.show()

