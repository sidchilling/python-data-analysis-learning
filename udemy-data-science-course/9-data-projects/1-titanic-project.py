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

# FacetGrids to show KDE plot for Ages for different Sexes
fig = sns.FacetGrid(df, hue = 'Sex', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
oldest = df['Age'].max()
fig.set(xlim = (0, oldest))
fig.add_legend()
plt.show()

# FacetGrid to show KDE plot for Person for different Sexes but showing histograms
fig = sns.FacetGrid(df, hue = 'Person', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.set(xlim = (0, oldest))
fig.add_legend()
plt.show()

# Age range by Pclass
fig = sns.FacetGrid(df, hue = 'Pclass', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.set(xlim = (0, oldest))
fig.add_legend()
plt.show()

## Question - What deck where the passengers were on and how it relates to class?

deck = df['Cabin'].dropna()
print deck.head()

def grab_deck_letter(deck):
    if deck is not np.NaN:
	return '{}'.format(deck[0])
    return np.NaN

deck = deck.apply(grab_deck_letter)
cabin_df = DataFrame(deck)
cabin_df = cabin_df[cabin_df['Cabin'] != 'T']

sns.factorplot(x = 'Cabin', data = cabin_df, palette = 'summer', kind = 'count')
plt.show()

# How does age vary for different decks

df['Deck'] = df['Cabin'].apply(grab_deck_letter)
df = df[df['Deck'] != 'T']
print df.head()
fig = sns.FacetGrid(df, hue = 'Deck', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.set(xlim = (0, oldest))
fig.add_legend()
plt.show()

sns.factorplot(x = 'Embarked', hue = 'Pclass', data = df, kind = 'count',
	      order = ['C', 'Q', 'S'])
plt.show()

# Who was alone and who were with family?
def was_with_family(family_members):
    siblings, parents_children = family_members
    siblings = int(siblings)
    parents_children = int(parents_children)

    if siblings > 0 or parents_children > 0:
	return 'With Family'
    return 'Without Family'

df['Family'] = df[['SibSp', 'Parch']].apply(was_with_family, axis = 1)
print df.head()
sns.factorplot(x = 'Family', data = df, kind = 'count')
plt.show()

# Which class has more families
sns.factorplot(x = 'Family', hue = 'Pclass', data = df, kind = 'count')
plt.show()

# What factors help someone survive the sinking of Titanic?
df['Survivor'] = df['Survived'].map({0 : 'No', 1 : 'Yes'})
print df.head()

sns.factorplot(x = 'Survivor', data = df, kind = 'count')
plt.show()

# Was class the factor for survival?
sns.factorplot(x = 'Pclass', y = 'Survived', data = df)
plt.show()

sns.factorplot(x = 'Pclass', y = 'Survived', hue = 'Person', data = df)
plt.show()

# whether age was a factor
sns.lmplot(x = 'Age', y = 'Survived', data = df)
plt.show()

# effect of class and age
sns.lmplot(x = 'Age', y = 'Survived', hue = 'Pclass', data = df,
	  palette = 'winter')
plt.show()

# Bin Ages
generations = [10, 20, 40, 60, 80]
sns.lmplot(x = 'Age', y = 'Survived', hue = 'Pclass', data = df,
	  palette = 'winter', x_bins = generations)
plt.show()

# How gender affects
sns.lmplot(x = 'Age', y = 'Survived', hue = 'Sex', data = df,
	  palette = 'winter', x_bins = generations)
plt.show()

# Did the deck have an effect on survival? And how does being a male affect this?
sns.factorplot(x = 'Deck', y = 'Survived', hue = 'Sex', data = df,
	      palette = 'winter', kind = 'bar')
plt.show()

# Did having a family member increase the survival?
sns.factorplot(x = 'Family', y = 'Survived', data = df, palette = 'winter',
	      kind = 'bar')
plt.show()

sns.factorplot(x = 'Family', y = 'Survived', hue = 'Sex', data = df, palette = 'winter',
	      kind = 'bar')
plt.show()
