import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('train.csv')
print df.head()

'''Columns description
1. PassengerId
2. Survived (1 for survived and 0 for not survived)
3. Pclass (which class)
4. Name
5. Sex
6. Age
7. SibSp (number of siblings)
8. Parch (number of parents / children)
9. Ticket
10. Fare
11. Cabin (contains many NaN values)
12. Embarked (cities they embarked from)
'''

print df.info()

# Answer the following questions - 
# Basic Questions - 
# 1. Who where the passengers on the Titanic? (Ages, Gender, Class... etc)
# 2. Where did the passengers come from?
# 3. Who was alone and who was with family?

# Advanced Question 
# 1. What factors helped someone survive the sinking?

# Show the distribution of sex (how many were males vs females)
# The best way will be to make a pie chart of all the genders
g = sns.factorplot(x = 'Sex', data = df, kind = 'count')
plt.subplots_adjust(top = 0.9)
g.fig.suptitle('Male vs Female (Absolute Numbers)')
plt.show()

# Show the percentage of male vs females
sex_dict = {
    'Sex' : ['Male', 'Female'],
    'Percents' : [round((len(df[df['Sex'] == 'male']) / float(len(df)) * 100), 2),
		  round((len(df[df['Sex'] == 'female']) / float(len(df)) * 100), 2)]
}
sex_df = DataFrame(sex_dict)
print sex_df

g = sns.factorplot(x = 'Sex', y = 'Percents', data = sex_df, kind = 'bar')
g.fig.suptitle('Male vs Female (Percents)')
plt.show()

# Compare sex according to class
classes_vs_sex_df = df.pivot_table(values = 'PassengerId', index = ['Sex'],
				  columns = ['Pclass'], aggfunc = 'count')
# Remove index to make a flat pivot table so that it can work with seaborn
classes_vs_sex_df = classes_vs_sex_df.reset_index()
print classes_vs_sex_df

sns.factorplot(x = 'Sex', data = df, hue = 'Pclass', kind = 'count')
plt.show()

sns.factorplot(x = 'Pclass', data = df, hue = 'Sex', kind = 'count')
plt.show()

# male vs female vs children
def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
	return 'Child'
    else:
	return sex.capitalize()
df['Person'] = df[['Age', 'Sex']].apply(male_female_child, axis = 1)
print df.head()

sns.factorplot(x = 'Person', data = df, kind = 'count')
plt.show()

sns.factorplot(x = 'Pclass', hue = 'Person', data = df, kind = 'count')
plt.show()

# Distribution of ages - Show a histogram
sns.distplot(a = df['Age'].dropna(), hist = True, kde = False, rug = False,
	    axlabel = 'Age', fit = stats.norm, bins = 70)
plt.show()

df['Age'].hist(bins = 70)
plt.show()

print 'Mean of Age: {}'.format(df['Age'].mean())
