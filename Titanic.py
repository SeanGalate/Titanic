# Titanic investigation
# What were the demographics of the passengers?
# What deck were the passengers on?
# Where did the passengers come from?
# Who was alone and who was with family?
# What factors help survives the sinking of the Titanic?

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Titanic csv file as a DataFrame
titanic_df = pd.read_csv('train.csv')

# Preview data
print(titanic_df.head(5))

# Preview info for the dataset
print(titanic_df.info())

# Graph passengers by sex
sns.catplot(x='Sex', kind='count', data=titanic_df)
plt.show()

# Graph genders by classes
sns.catplot(x='Sex', hue='Pclass', kind='count', data=titanic_df)
plt.show()

# Function to categorize male adults, female adults and children


def male_female_child(passenger):
    age, sex = passenger

    if age < 16:
        return 'child'
    else:
        return sex

# Add new column called person to DataFrame


titanic_df['person'] = titanic_df['Age', 'Sex'].apply(male_female_child, axis=1)
print(titanic_df.head(10))

# Graph persons by class
sns.catplot(x='Pclass', hue='person', kind='count', data=titanic_df)
plt.show()

# Histogram by age
titanic_df['Age'].hist(bins=70)
plt.show()

# Total for each category of person
print(titanic_df['person'].value_counts())

# What were the demographics of the passengers?

# Set figure equal to a facetgrid with the pandas DataFrame as its data source
fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)

# Plot kdeplots for age by hue choice
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
plt.show()

# Repeat process but for person
fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
plt.show()

# Repeat the same process for class by changing hue
fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
plt.show()

# What deck were the passengers on?

# Create deck object and drop NaN values
deck = titanic_df['Cabin'].dropna()
print(deck.head(5))

# Loop to acquire the letter for the deck level
levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot(x='Cabin', kind='count', data=cabin_df, palette='winter_d')
plt.show()

# Where did the passengers come from?

# Graph showing embarked by class
sns.catplot(x='Embarked', kind='count', data=titanic_df, hue='Pclass', order=['Q', 'C', 'S'])
plt.show()

# Who was alone and who was with family?

# Add column to DataFrame to define Alone
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
print(titanic_df['Alone'])

# Criteria for Alone status
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
print(titanic_df.head(5))

# Graph of the number Alone and With Family
sns.catplot(x='Alone', kind='count', data=titanic_df, palette='Blues')
plt.show()

# What factors help survives the sinking of the Titanic?

# Add new column to the DataFrame regarding survivors
titanic_df['Survivor'] = titanic_df.Survived.map({0: 'no', 1: 'yes'})

# Graph showing the number of survivors
sns.catplot(x='Survivor', kind='count', data=titanic_df, palette='Set1')
plt.show()

# Graph showing survivors by class
sns.catplot(x='Pclass', y='Survived', kind='point', hue='person', data=titanic_df)
plt.show()

# Graph showing survivors by class and person
sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_df, palette='winter')
plt.show()

# Linear plot on age vs survival
generations = [10, 20, 40, 60, 80]
sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_df, x_bins=generations)
plt.show()

# Linear plot on age vs survival using hue to separate class
sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df, palette='winter', x_bins=generations)
plt.show()
