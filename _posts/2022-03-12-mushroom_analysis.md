# Of Magic and Mushrooms



## Introduction and Importing


WOW!! A blog?! Exciting! Hello and welcome, for my first trick I'll be exploring [a Synthetic Mushroom dataset](https://www.kaggle.com/uciml/mushroom-classification) from Kaggle. This dataset contains 8124 synthetic datapoints corresponding to 23 species of gilled mushrooms from the genera Agaricus and Lepiota, classified into edible (definitely safe to eat) or poisonous (of uncertain toxicity). I'm just gonna load it up and work my way through it, hopefully an interesting avenue of investigation presents itself.  

Table of Contents:
- [Initial Look and Pre-Processing](#"init-look")
- [EDA](#"EDA")
	- [Round 1](#id "round1")
	- [Missing](#id "mv")
- [Initial Modelling](#id"im")
    - [K-Nearest-Neighbors](#id"knn")
    - [RandomForests and Decision Trees](#id"rf")
- [Feature Importance](#id"fi")
- [Feature Reduction](#id"fr")
    - [Challenge Mode](#id "chal1")
    - [Challenge Mode Round 2](#id "chal2")
- [Bonus Round](#id"c3")

No prizes for guessing what comes first. First step ( as always) is to import our required modules. For this project I was tasked with using KNN, so it's first on the list. I'm also lazy and want some quick and easy feature importance so I've gone with Decision Tree and RandomForest classifiers too.

Given that the task is class identification, I anticipate needing more than just accuracy, so I'm also grabbing a selection of metrics... 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import math
from itertools import permutations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_validate

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
```

Ahh my arch-nemesis; importing files. What did I save the csv as...



```python
os.listdir()
```




    ['.ipynb_checkpoints',
     'mushrooms.csv',
     'mushroom_analysis.ipynb',
     'mushroom_missing_split.ipynb',
     'mushroom_species_finder.ipynb',
     'mushroom_stalk_type.ipynb',
     'plots_for_antonis.ipynb']



That was better than my usual naming convention. Let's read it in and have a little look shall we?

```python
mushroom=pd.read_csv('mushrooms.csv')
```

```python
mushroom.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



# Initial Look and Pre-Processing <a name="init-look"></a>

This looks very confusing. Do we have a data dictionary?

```python

lazy="""
    cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

    cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

    cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

    bruises: yes=t,no=f

    odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

    gill-attachment: attached=a,descending=d,free=f,notched=n

    gill-spacing: close=c,crowded=w,distant=d

    gill-size: broad=b,narrow=n

    gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

    stalk-shape: enlarging=e,tapering=t

    stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

    stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

    stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

    stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

    stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

    veil-type: partial=p,universal=u

    veil-color: brown=n,orange=o,white=w,yellow=y

    ring-number: none=n,one=o,two=t

    ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

    spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

    population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

    habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
    
    """

```

I mean I guess we do legally, but this would be a pain to set by hand. Let's see if we can split this pythonically...

```python
lazy=lazy.split('\n\n')
```

```python
for i in range(0,3):
    print(lazy[i])
```

    
        cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
        cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
        cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
    

Ok, this looks promising!! To map this nicely we'll need a dictionary. Can we split this into keys and values?? Maybe if we split on the colon?

```python
for line in lazy:
    key=line.split(':')[0]
    vals=line.split(':')[1]
    print(key)
    print(vals)
```

    
        cap-shape
     bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
        cap-surface
     fibrous=f,grooves=g,scaly=y,smooth=s
        cap-color
     brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
        bruises
     yes=t,no=f
        odor
     almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
        gill-attachment
     attached=a,descending=d,free=f,notched=n
        gill-spacing
     close=c,crowded=w,distant=d
        gill-size
     broad=b,narrow=n
        gill-color
     black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
        stalk-shape
     enlarging=e,tapering=t
        stalk-root
     bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
        stalk-surface-above-ring
     fibrous=f,scaly=y,silky=k,smooth=s
        stalk-surface-below-ring
     fibrous=f,scaly=y,silky=k,smooth=s
        stalk-color-above-ring
     brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
        stalk-color-below-ring
     brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
        veil-type
     partial=p,universal=u
        veil-color
     brown=n,orange=o,white=w,yellow=y
        ring-number
     none=n,one=o,two=t
        ring-type
     cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
        spore-print-color
     black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
        population
     abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
        habitat
     grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
        
        
    

Looks like we have trailing spaces, let's try again

```python
for line in lazy:
    key=line.split(':')[0].strip()
    vals=line.split(':')[1].strip()
    print(key)
    print(vals)
```

    cap-shape
    bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
    cap-surface
    fibrous=f,grooves=g,scaly=y,smooth=s
    cap-color
    brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
    bruises
    yes=t,no=f
    odor
    almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
    gill-attachment
    attached=a,descending=d,free=f,notched=n
    gill-spacing
    close=c,crowded=w,distant=d
    gill-size
    broad=b,narrow=n
    gill-color
    black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
    stalk-shape
    enlarging=e,tapering=t
    stalk-root
    bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
    stalk-surface-above-ring
    fibrous=f,scaly=y,silky=k,smooth=s
    stalk-surface-below-ring
    fibrous=f,scaly=y,silky=k,smooth=s
    stalk-color-above-ring
    brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
    stalk-color-below-ring
    brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
    veil-type
    partial=p,universal=u
    veil-color
    brown=n,orange=o,white=w,yellow=y
    ring-number
    none=n,one=o,two=t
    ring-type
    cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
    spore-print-color
    black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
    population
    abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
    habitat
    grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
    

Looks better to me!! But we can't use this for a dictionary.... We'll need to split val again, into indiviual items that we can compare our data against. 

```python
for line in lazy:
    key=line.split(':')[0].strip()
    vals=line.split(':')[1].strip()
    vals=vals.split(',')
    print(vals)
    print(vals[0])
```

    ['bell=b', 'conical=c', 'convex=x', 'flat=f', ' knobbed=k', 'sunken=s']
    bell=b
    ['fibrous=f', 'grooves=g', 'scaly=y', 'smooth=s']
    fibrous=f
    ['brown=n', 'buff=b', 'cinnamon=c', 'gray=g', 'green=r', 'pink=p', 'purple=u', 'red=e', 'white=w', 'yellow=y']
    brown=n
    ['yes=t', 'no=f']
    yes=t
    ['almond=a', 'anise=l', 'creosote=c', 'fishy=y', 'foul=f', 'musty=m', 'none=n', 'pungent=p', 'spicy=s']
    almond=a
    ['attached=a', 'descending=d', 'free=f', 'notched=n']
    attached=a
    ['close=c', 'crowded=w', 'distant=d']
    close=c
    ['broad=b', 'narrow=n']
    broad=b
    ['black=k', 'brown=n', 'buff=b', 'chocolate=h', 'gray=g', ' green=r', 'orange=o', 'pink=p', 'purple=u', 'red=e', 'white=w', 'yellow=y']
    black=k
    ['enlarging=e', 'tapering=t']
    enlarging=e
    ['bulbous=b', 'club=c', 'cup=u', 'equal=e', 'rhizomorphs=z', 'rooted=r', 'missing=?']
    bulbous=b
    ['fibrous=f', 'scaly=y', 'silky=k', 'smooth=s']
    fibrous=f
    ['fibrous=f', 'scaly=y', 'silky=k', 'smooth=s']
    fibrous=f
    ['brown=n', 'buff=b', 'cinnamon=c', 'gray=g', 'orange=o', 'pink=p', 'red=e', 'white=w', 'yellow=y']
    brown=n
    ['brown=n', 'buff=b', 'cinnamon=c', 'gray=g', 'orange=o', 'pink=p', 'red=e', 'white=w', 'yellow=y']
    brown=n
    ['partial=p', 'universal=u']
    partial=p
    ['brown=n', 'orange=o', 'white=w', 'yellow=y']
    brown=n
    ['none=n', 'one=o', 'two=t']
    none=n
    ['cobwebby=c', 'evanescent=e', 'flaring=f', 'large=l', 'none=n', 'pendant=p', 'sheathing=s', 'zone=z']
    cobwebby=c
    ['black=k', 'brown=n', 'buff=b', 'chocolate=h', 'green=r', 'orange=o', 'purple=u', 'white=w', 'yellow=y']
    black=k
    ['abundant=a', 'clustered=c', 'numerous=n', 'scattered=s', 'several=v', 'solitary=y']
    abundant=a
    ['grasses=g', 'leaves=l', 'meadows=m', 'paths=p', 'urban=u', 'waste=w', 'woods=d']
    grasses=g
    

The fact that I can index val means that we've split it into individual items. We're so close to a dictionary!! We really want to swap the values around though. (We want e.g 'b':'bell, 'c':'conical' etc.) Maybe if we split and build a dictionary?

Let's try on the last set of vals

```python
print(vals)
```

    ['grasses=g', 'leaves=l', 'meadows=m', 'paths=p', 'urban=u', 'waste=w', 'woods=d']
    

```python
vals={i.split('=')[1]:i.split('=')[0] for i in vals}
print(vals)
```

    {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}
    

```python
type(vals)
```




    dict



We've nailed it!! Now we need to build the full dictionary

```python
feat_dict={}
for line in lazy:
    key=line.split(':')[0].strip()
    vals=line.split(':')[1].strip()
    vals=vals.split(',')
    vals={i.split('=')[1]:i.split('=')[0] for i in vals}
    feat_dict.update({key:vals})
```

```python
feat_dict['cap-shape']
```




    {'b': 'bell',
     'c': 'conical',
     'x': 'convex',
     'f': 'flat',
     'k': ' knobbed',
     's': 'sunken'}



```python
feat_dict['habitat']
```




    {'g': 'grasses',
     'l': 'leaves',
     'm': 'meadows',
     'p': 'paths',
     'u': 'urban',
     'w': 'waste',
     'd': 'woods'}



```python
feat_dict['habitat']['g']
```




    'grasses'



Looks like a dictionary to me, shall we see if it works?!?

```python
test=mushroom.copy()
test=test.replace(feat_dict)
```

```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>convex</td>
      <td>smooth</td>
      <td>brown</td>
      <td>yes</td>
      <td>pungent</td>
      <td>free</td>
      <td>close</td>
      <td>narrow</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>black</td>
      <td>scattered</td>
      <td>urban</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>convex</td>
      <td>smooth</td>
      <td>yellow</td>
      <td>yes</td>
      <td>almond</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>brown</td>
      <td>numerous</td>
      <td>grasses</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>bell</td>
      <td>smooth</td>
      <td>white</td>
      <td>yes</td>
      <td>anise</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>brown</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>brown</td>
      <td>numerous</td>
      <td>meadows</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>convex</td>
      <td>scaly</td>
      <td>white</td>
      <td>yes</td>
      <td>pungent</td>
      <td>free</td>
      <td>close</td>
      <td>narrow</td>
      <td>brown</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>black</td>
      <td>scattered</td>
      <td>urban</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>convex</td>
      <td>smooth</td>
      <td>gray</td>
      <td>no</td>
      <td>none</td>
      <td>free</td>
      <td>crowded</td>
      <td>broad</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>evanescent</td>
      <td>brown</td>
      <td>abundant</td>
      <td>grasses</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



That looks like what we want. Let's transform our data!

```python
mushroom=mushroom.replace(feat_dict)
```

As a first point of call I like to take a little look at my data, see what's around, what's missing, and what sort of values I'm working with. Most of this is encapsulated by df.info() and df.describe(), but I've written a function to give it a little twist...

```python
def key_stats(df):
    """
    Take a dataframe (df) and return its datatypes, 
    column counts and number of null values as 
    a new dataframe
    """
    dtypes=df.dtypes
    counts=df.count()
    nulls=df.isna().sum()
    stats=[dtypes,counts,nulls]
    wow=pd.concat(stats,axis=1).reset_index().rename(columns={'index':'feature',0:'dtypes',1:'count',2:"na's"})
    wow['unique']=wow['feature'].apply(lambda x: df[x].unique())
    wow['unique count']=wow['unique'].apply(lambda x: len(x))
    return wow
```

```python
stats=key_stats(mushroom)
stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>dtypes</th>
      <th>count</th>
      <th>na's</th>
      <th>unique</th>
      <th>unique count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>class</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[p, e]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cap-shape</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[convex, bell, sunken, flat,  knobbed, conical]</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cap-surface</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[smooth, scaly, fibrous, grooves]</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cap-color</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[brown, yellow, white, gray, red, pink, buff, ...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bruises</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[yes, no]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>odor</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[pungent, almond, anise, none, foul, creosote,...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>gill-attachment</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[free, attached]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>gill-spacing</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[close, crowded]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gill-size</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[narrow, broad]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>gill-color</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[black, brown, gray, pink, white, chocolate, p...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>stalk-shape</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[enlarging, tapering]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>stalk-root</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[equal, club, bulbous, rooted, missing]</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>stalk-surface-above-ring</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[smooth, fibrous, silky, scaly]</td>
      <td>4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>stalk-surface-below-ring</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[smooth, fibrous, scaly, silky]</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>stalk-color-above-ring</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[white, gray, pink, brown, buff, red, orange, ...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>15</th>
      <td>stalk-color-below-ring</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[white, pink, gray, buff, brown, red, yellow, ...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>veil-type</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[partial]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>veil-color</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[white, brown, orange, yellow]</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ring-number</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[one, two, none]</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ring-type</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[pendant, evanescent, large, flaring, none]</td>
      <td>5</td>
    </tr>
    <tr>
      <th>20</th>
      <td>spore-print-color</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[black, brown, purple, chocolate, white, green...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>21</th>
      <td>population</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[scattered, numerous, abundant, several, solit...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>22</th>
      <td>habitat</td>
      <td>object</td>
      <td>8124</td>
      <td>0</td>
      <td>[urban, grasses, meadows, woods, paths, waste,...</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



From the table we can see a couple of key points:
- All the data is categorical, we may have to one hot encode...
- Veil-type only has one value, no need to keep it
- There's a lot of variation in the dimensions of features (some have 9 values, others have 2); scaling/proper encoding may be important here.
- Nothing seems to be missing

```python
mushroom.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>...</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>...</td>
      <td>4</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>e</td>
      <td>convex</td>
      <td>scaly</td>
      <td>brown</td>
      <td>no</td>
      <td>none</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>buff</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>white</td>
      <td>several</td>
      <td>woods</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4208</td>
      <td>3656</td>
      <td>3244</td>
      <td>2284</td>
      <td>4748</td>
      <td>3528</td>
      <td>7914</td>
      <td>6812</td>
      <td>5612</td>
      <td>1728</td>
      <td>...</td>
      <td>4936</td>
      <td>4464</td>
      <td>4384</td>
      <td>8124</td>
      <td>7924</td>
      <td>7488</td>
      <td>3968</td>
      <td>2388</td>
      <td>4040</td>
      <td>3148</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 23 columns</p>
</div>



```python
mushroom=mushroom.drop(columns=['veil-type'])
```

# EDA  <a name="EDA"></a>


## Round 1 <a name="round1"></a>
Let's try some EDA.
Just as an initial go, let's look at the distribution of odour across the two classes

```python
plt.figure()
sns.catplot(data=mushroom,x='odor',hue='class',kind='count')
plt.xticks(rotation=45, ha='right')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
![png](/images/mushroom_analysis_files/output_43_1.png)
    


Very clear split here. Pungent, foul, fishy, spicy and musty mushrooms are all poisonous

```python
plt.figure(figsize=(10,10))
sns.catplot(data=mushroom,x='cap-color',hue='class',kind='count')
plt.xticks(rotation=45,ha='right')
plt.show()
```


    <Figure size 720x720 with 0 Axes>



    
![png](/images/mushroom_analysis_files/output_45_1.png)
    


Less clear split here. But it appears purple and green mushrooms are all poisonous.
Let's plot the distribution  of features across classes

```python
def plotter(data,hue,n_cols,fig_size):
    n_features=len(data.columns)
    n_rows=math.ceil(n_features/n_cols)
    fig,axs=plt.subplots(n_rows,n_cols,figsize=fig_size)
    for i,j in enumerate(data.columns):
        ax=axs[math.floor(i/n_cols),i%n_cols]
        sns.countplot(data=data,x=j,hue=hue,ax=ax)
        ax.set_xlabel(f"{j}",fontsize=25)
        ax.set_ylabel("Count",fontsize=25)
        ax.set_xticklabels(labels=data[j].unique(),rotation=45, ha='right',fontsize=25)
        #ax.set_yticks
        ax.set_yticklabels(labels=ax.get_yticklabels(minor=True),rotation=45, ha='right',fontsize=25)
        ax.set_title(f'Breakdown of {j} by {hue}',fontsize=30)
        ax.legend(fontsize=20)
    plt.tight_layout()
    plt.show()
```

```python
plotter(mushroom,'class',3,(30,70))
```


    
![png](/images/mushroom_analysis_files/output_48_0.png)
    


OK, sure that's gnarly, but we've got some takeaways.
- Looks like the classes are balanced
- Quite a few features are exclusive to poisonous or edible species. 
    - As discussed earlier, lot's of smells are exclusive to poisonous or edible varieties
    - All mushrooms with buff coloured gills are poisonous 
    - All mushrooms with large rings are poisonous
- Other features aren't exclusive, but give overwhelming odds
    - Most mushrooms that don't bruise are poisonous
    - A lot of white or chocolate coloured spore print mushrooms are poinsonous
    
- We're expecting good results...

## Missing values<a name="mv"></a>

Re-Re-Wind!! What are the values for stalk root?!?!

```python
feat_dict['stalk-root']
```




    {'b': 'bulbous',
     'c': 'club',
     'u': 'cup',
     'e': 'equal',
     'z': 'rhizomorphs',
     'r': 'rooted',
     '?': 'missing'}



missing?!? That's not good!
How many are there?

```python
miss=len(mushroom[mushroom['stalk-root']=='missing'])
mush=len(mushroom)
print(f'{(miss/mush)*100:.2f}% shrooms have a missing root type') 
print(f'We have data on {mush} mushrooms') 
```

    30.53% shrooms have a missing root type
    We have data on 8124 mushrooms
    

Hmmmm 30%? That's like, a lot. We'd drop a lot of data if we just removed them.
Dare we hope that they tell us something?
Let's look at the disribution and hope that there's some meaning here.

```python
missers=mushroom[mushroom['stalk-root']=='missing'].copy()
present=mushroom[mushroom['stalk-root']!='missing'].copy()
n_feats=len(mushroom.columns)
fig,axs=plt.subplots(n_feats,2,figsize=(10,80))
for i,col in enumerate(mushroom.columns):
    ax=axs[i,0]
    sns.countplot(data=missers,x=col,ax=ax,palette='tab10',order=mushroom[col].unique())
    ax.set_title(f'Distribution of {col} for missing roots')
    if len(mushroom[col].unique())>5:
        ax.set_xticklabels(labels=mushroom[col].unique(),rotation=45, ha='right')
    ax=axs[i,1]
    sns.countplot(data=present,x=col,ax=ax,palette='tab10',order=mushroom[col].unique())
    ax.set_title(f'Distribution of {col} for other root types')
    if len(mushroom[col].unique())>5:
        ax.set_xticklabels(labels=mushroom[col].unique(),rotation=45, ha='right')
plt.tight_layout()
plt.show()
```


    
![png](/images/mushroom_analysis_files/output_55_0.png)
    


Some of these feature distributions are massively different. Not a mushroom-ologist or anything, but maybe there's some significance to the stalk root being missing?

Let's not drop it for now...

# Initial Modelling<a name="im"></a>

For my first point of call, I got thinking. Given the amount of data I'm about to one-hot-encode, is there even a sensible distance metric to try? Using something like Euclidean distance gives a bounded max distance of $\sqrt{2}$ along any 2 axes in the feature space...

Well, if one-hot-encoding takes inspiration from the world of bits, maybe I will too. I thought to try using the hamming distance ( a binary distance metric) that simply encodes the distance between two points as the number of bits that are different. This would give us a similarity scale ranging from 0-21 (one bit difference per feature) to play with. 

With that in mind I used pandas to one hot encode my data, and implemented a simple train-test-split and fit a simple K-Nearest-Neighbors model to the data. I also display the full classification report. 

```python
X=mushroom.drop(columns='class')
y=mushroom['class']
```

```python
X=pd.get_dummies(X)
```

```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cap-shape_ knobbed</th>
      <th>cap-shape_bell</th>
      <th>cap-shape_conical</th>
      <th>cap-shape_convex</th>
      <th>cap-shape_flat</th>
      <th>cap-shape_sunken</th>
      <th>cap-surface_fibrous</th>
      <th>cap-surface_grooves</th>
      <th>cap-surface_scaly</th>
      <th>cap-surface_smooth</th>
      <th>...</th>
      <th>population_scattered</th>
      <th>population_several</th>
      <th>population_solitary</th>
      <th>habitat_grasses</th>
      <th>habitat_leaves</th>
      <th>habitat_meadows</th>
      <th>habitat_paths</th>
      <th>habitat_urban</th>
      <th>habitat_waste</th>
      <th>habitat_woods</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 116 columns</p>
</div>



## K Nearest Neighbors<a name="KNN"></a>

```python
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
```

```python
knn=KNeighborsClassifier(n_neighbors=10, metric='hamming').fit(X_train,y_train)
preds=knn.predict(X_train)
```

```python
print(classification_report(y_train,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      1.00      1.00      3365
               p       1.00      1.00      1.00      3134
    
        accuracy                           1.00      6499
       macro avg       1.00      1.00      1.00      6499
    weighted avg       1.00      1.00      1.00      6499
    
    

```python
preds=knn.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      1.00      1.00       843
               p       1.00      1.00      1.00       782
    
        accuracy                           1.00      1625
       macro avg       1.00      1.00      1.00      1625
    weighted avg       1.00      1.00      1.00      1625
    
    

These results look good, too good. Now I'm suspicious I messed up somewhere in the process. 

It's super not scientific, but if I can ruin my perfect score I'd feel a bit better about whether or not I've messed up. 
Can I decrease the accuracy using hyperparameters?

```python
knn=KNeighborsClassifier(n_neighbors=100, metric='hamming').fit(X_train,y_train)
preds=knn.predict(X_train)
```

```python
print(classification_report(y_train,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      0.98      0.99      3365
               p       0.98      1.00      0.99      3134
    
        accuracy                           0.99      6499
       macro avg       0.99      0.99      0.99      6499
    weighted avg       0.99      0.99      0.99      6499
    
    

```python
preds=knn.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      0.98      0.99       843
               p       0.98      1.00      0.99       782
    
        accuracy                           0.99      1625
       macro avg       0.99      0.99      0.99      1625
    weighted avg       0.99      0.99      0.99      1625
    
    

How about a silly distance metric?

```python
knn=KNeighborsClassifier(n_neighbors=10, metric='chebyshev').fit(X_train,y_train)
preds=knn.predict(X_train)
```

```python
print(classification_report(y_train,preds))
```

                  precision    recall  f1-score   support
    
               e       0.47      0.49      0.48      3365
               p       0.42      0.40      0.41      3134
    
        accuracy                           0.45      6499
       macro avg       0.44      0.45      0.44      6499
    weighted avg       0.45      0.45      0.45      6499
    
    

```python
preds=knn.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       0.47      0.51      0.49       843
               p       0.42      0.39      0.41       782
    
        accuracy                           0.45      1625
       macro avg       0.45      0.45      0.45      1625
    weighted avg       0.45      0.45      0.45      1625
    
    

Ok, using 100 neighbours or a non-applicable metric it's possible for the model to be wrong. Looks like the data is just very kind to us!

Mushroom-ologists, you might be out of work pretty darn soon...

## Random Forests and Decision Trees<a name="rf"></a>

Ok, KNN is cool, but what if I want some feature importance? Mushrooms are like, forrest-y no?
Let's try some trees (Decision Trees and Random Forests). As above, except I fit a RandomForest and Decision Tree classifer to the same train test split I implemented for KNN (consistency is key my dudes). Again we're looking at the whole classification report here, to try and understand what the models are good and bad at.

```python
rf=RandomForestClassifier(n_estimators=10,random_state=42).fit(X_train,y_train)
preds=rf.predict(X_train)
print(classification_report(y_train,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      1.00      1.00      3365
               p       1.00      1.00      1.00      3134
    
        accuracy                           1.00      6499
       macro avg       1.00      1.00      1.00      6499
    weighted avg       1.00      1.00      1.00      6499
    
    

```python
preds=rf.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      1.00      1.00       843
               p       1.00      1.00      1.00       782
    
        accuracy                           1.00      1625
       macro avg       1.00      1.00      1.00      1625
    weighted avg       1.00      1.00      1.00      1625
    
    

```python
dt=DecisionTreeClassifier(criterion='entropy').fit(X_train,y_train)
preds=dt.predict(X_train)
print(classification_report(y_train,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      1.00      1.00      3365
               p       1.00      1.00      1.00      3134
    
        accuracy                           1.00      6499
       macro avg       1.00      1.00      1.00      6499
    weighted avg       1.00      1.00      1.00      6499
    
    

```python
preds=dt.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      1.00      1.00       843
               p       1.00      1.00      1.00       782
    
        accuracy                           1.00      1625
       macro avg       1.00      1.00      1.00      1625
    weighted avg       1.00      1.00      1.00      1625
    
    

## Feature Importance <a name="fi"></a>

Okay, that went a bit too well as well. Hmm this is going swimmmingly huh? But, using these trees, we can have a look at feature importance.

```python
importances = rf.feature_importances_
std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis=0)
forest_importances = pd.DataFrame(importances, index=X_train.columns)
forest_importances.rename(columns={0:'MDI'},inplace=True)
forest_importances.sort_values('MDI',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>odor_none</th>
      <td>0.128677</td>
    </tr>
    <tr>
      <th>gill-size_narrow</th>
      <td>0.105161</td>
    </tr>
    <tr>
      <th>ring-type_pendant</th>
      <td>0.064620</td>
    </tr>
    <tr>
      <th>stalk-surface-above-ring_silky</th>
      <td>0.060745</td>
    </tr>
    <tr>
      <th>ring-type_large</th>
      <td>0.058291</td>
    </tr>
    <tr>
      <th>spore-print-color_chocolate</th>
      <td>0.054278</td>
    </tr>
    <tr>
      <th>gill-color_buff</th>
      <td>0.039359</td>
    </tr>
    <tr>
      <th>stalk-surface-below-ring_silky</th>
      <td>0.038696</td>
    </tr>
    <tr>
      <th>odor_creosote</th>
      <td>0.023358</td>
    </tr>
    <tr>
      <th>stalk-color-below-ring_white</th>
      <td>0.023003</td>
    </tr>
  </tbody>
</table>
</div>



Random Forests are well, random, let's run a few more and look at some modally important features.
If I generate the above table for several runs of the classifier, I can combine them into one dataframe, and look at how often a feature appears in say the top 10, to get an idea of which features end up being important

```python
def importance_group(X,y,n):
    dfs=[]
    for i in range(n):
        rf=RandomForestClassifier(n_estimators=n).fit(X,y)
        importances = rf.feature_importances_
        forest_importances = pd.DataFrame(importances, index=X_train.columns)
        a=forest_importances.sort_values(0,ascending=False).head(10)
        dfs.append(a)
    dfs=pd.concat(dfs,axis=0).reset_index().rename(columns={'index':'feature',0:'MDI'})
    f_counts=dfs.groupby('feature')[['MDI']].count().sort_values('MDI',ascending=False)
    f_counts['Percent']=f_counts['MDI'].apply(lambda x: (x/n)*100)
    return f_counts
```

```python
c=importance_group(X_train,y_train,50)
c
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDI</th>
      <th>Percent</th>
    </tr>
    <tr>
      <th>feature</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>odor_none</th>
      <td>50</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>gill-size_narrow</th>
      <td>49</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>odor_foul</th>
      <td>49</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>gill-size_broad</th>
      <td>49</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>stalk-surface-above-ring_silky</th>
      <td>45</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>spore-print-color_chocolate</th>
      <td>44</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>stalk-surface-below-ring_silky</th>
      <td>44</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>gill-color_buff</th>
      <td>43</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>ring-type_pendant</th>
      <td>39</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>bruises_yes</th>
      <td>27</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>bruises_no</th>
      <td>16</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>ring-type_large</th>
      <td>12</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>stalk-surface-above-ring_smooth</th>
      <td>11</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>population_several</th>
      <td>9</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>gill-spacing_close</th>
      <td>4</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>gill-spacing_crowded</th>
      <td>4</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>stalk-root_bulbous</th>
      <td>2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>ring-type_evanescent</th>
      <td>1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>stalk-root_equal</th>
      <td>1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>stalk-surface-below-ring_smooth</th>
      <td>1</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(dt, max_depth=None, feature_names=X_train.columns, class_names=['e','p'], label='all',filled=True, impurity=True)
plt.show()
```


    
![png](/images/mushroom_analysis_files/output_86_0.png)
    


Ok, by kinda cross-referencing the rf and dt importances odour looks like a really strong predictor across the board (sounds about right if we remember our EDA), and spore print colour holds its own as well.


# Feature  Reduction<a name="fr"></a>

## Challenge Mode!!<a name="chal1"></a>

How well can I do using just odour?
Second verse, same as the first!!

```python
X=mushroom[['odor']]
y=mushroom['class']
```

```python
X=pd.get_dummies(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
```

```python
knn=KNeighborsClassifier(n_neighbors=10, metric='hamming').fit(X_train,y_train)
preds=knn.predict(X_train)
print(classification_report(y_train,preds))
preds=knn.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       0.97      1.00      0.99      3365
               p       1.00      0.97      0.98      3134
    
        accuracy                           0.99      6499
       macro avg       0.99      0.98      0.99      6499
    weighted avg       0.99      0.99      0.99      6499
    
                  precision    recall  f1-score   support
    
               e       0.97      1.00      0.99       843
               p       1.00      0.97      0.98       782
    
        accuracy                           0.98      1625
       macro avg       0.99      0.98      0.98      1625
    weighted avg       0.99      0.98      0.98      1625
    
    

```python
dt=DecisionTreeClassifier(criterion='entropy').fit(X_train,y_train)
preds=dt.predict(X_train)
print(classification_report(y_train,preds))
preds=dt.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       0.97      1.00      0.99      3365
               p       1.00      0.97      0.98      3134
    
        accuracy                           0.99      6499
       macro avg       0.99      0.98      0.99      6499
    weighted avg       0.99      0.99      0.99      6499
    
                  precision    recall  f1-score   support
    
               e       0.97      1.00      0.99       843
               p       1.00      0.97      0.98       782
    
        accuracy                           0.98      1625
       macro avg       0.99      0.98      0.98      1625
    weighted avg       0.99      0.98      0.98      1625
    
    

```python
plt.figure(figsize=(20,20))
tree.plot_tree(dt, max_depth=None, feature_names=X_train.columns, class_names=['e','p'], label='all',filled=True, impurity=True)
plt.show()
```


    
![png](/images/mushroom_analysis_files/output_94_0.png)
    


Now that there's a very real chance the classification won't be perfect, I'm going to focus on poisonous recall: I can live (literally) with miss-classifying an edible mushroom as poisonous (I just miss out on its shroomy goodness). However, no amount of shroom-based happpiness saves me if I eat a poinsous variety I thought was edible. 

Case in point: Using the decision tree there is a chance you eat one of the 95 poisonous mushrooms the tree thinks are totally fine.

To our credit though we still did pretty well. An F1 score of 0.98 using one feature is pretty crazy! 

So how many features do I need to get a perfect score?
Third verse, same as the first (and second...)

Except this time I added a *Slight Twist$^{TM}$*. Starting with just odor, I'll find the pair of features that give the best recall. I'll then add features stepwise, maximising recall each time to find the least amount of features that let me eat mushrooms safely.  

```python
def combo_finder(data,fixed,target):
    tests=pd.DataFrame(columns=['cols','accuracy','precision','recall','f1'])
    X=data.drop(columns=[target])
    y=data[target]
    y=y.apply(lambda x:1 if x=='p' else 0)
    fl=list(X.columns)
    for i in fixed:
        fl.remove(i)
    for i,col in enumerate(fl):
        new_fl=fixed+[col]
        X=data[new_fl]
        X=pd.get_dummies(X)
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
        knn=KNeighborsClassifier(n_neighbors=10, metric='hamming').fit(X_train,y_train)
        preds=knn.predict(X_train)
        acc=accuracy_score(y_train,preds)
        pres=precision_score(y_train,preds)
        rec=recall_score(y_train,preds)
        f1=f1_score(y_train,preds)
        tests.loc[i,'cols']=new_fl
        tests.loc[i,'accuracy']=acc
        tests.loc[i,'precision']=pres
        tests.loc[i,'recall']=rec
        tests.loc[i,'f1']=f1
    return tests
```

```python
experiments=combo_finder(mushroom,['odor'],'class')
```

```python
experiments.sort_values('recall',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>[odor, spore-print-color]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[odor, stalk-surface-below-ring]</td>
      <td>0.987998</td>
      <td>0.995782</td>
      <td>0.97926</td>
      <td>0.987452</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[odor, habitat]</td>
      <td>0.989845</td>
      <td>1</td>
      <td>0.978941</td>
      <td>0.989358</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[odor, cap-color]</td>
      <td>0.988306</td>
      <td>1</td>
      <td>0.97575</td>
      <td>0.987726</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[odor, gill-color]</td>
      <td>0.988152</td>
      <td>1</td>
      <td>0.975431</td>
      <td>0.987563</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[odor, stalk-color-below-ring]</td>
      <td>0.987998</td>
      <td>1</td>
      <td>0.975112</td>
      <td>0.987399</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[odor, stalk-color-above-ring]</td>
      <td>0.986459</td>
      <td>1</td>
      <td>0.971921</td>
      <td>0.985761</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[odor, veil-color]</td>
      <td>0.986459</td>
      <td>1</td>
      <td>0.971921</td>
      <td>0.985761</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[odor, stalk-root]</td>
      <td>0.986459</td>
      <td>1</td>
      <td>0.971921</td>
      <td>0.985761</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[odor, population]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[odor, ring-type]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[odor, ring-number]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[odor, cap-shape]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[odor, cap-surface]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[odor, stalk-shape]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[odor, gill-size]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[odor, gill-spacing]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[odor, gill-attachment]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[odor, bruises]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[odor, stalk-surface-above-ring]</td>
      <td>0.985382</td>
      <td>1</td>
      <td>0.969687</td>
      <td>0.98461</td>
    </tr>
  </tbody>
</table>
</div>



98.8% Recall score using odor and spore-print-color

```python
experiments=combo_finder(mushroom,['odor','spore-print-color'],'class')
```

```python
experiments.sort_values('recall',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>[odor, spore-print-color, ring-number]</td>
      <td>0.994461</td>
      <td>0.988644</td>
      <td>1</td>
      <td>0.994289</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[odor, spore-print-color, gill-size]</td>
      <td>0.994461</td>
      <td>0.988644</td>
      <td>1</td>
      <td>0.994289</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[odor, spore-print-color, stalk-surface-below-...</td>
      <td>0.996923</td>
      <td>0.99586</td>
      <td>0.997766</td>
      <td>0.996812</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[odor, spore-print-color, habitat]</td>
      <td>0.996769</td>
      <td>0.997761</td>
      <td>0.995533</td>
      <td>0.996646</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[odor, spore-print-color, cap-color]</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[odor, spore-print-color, stalk-color-below-ring]</td>
      <td>0.996923</td>
      <td>1</td>
      <td>0.993618</td>
      <td>0.996799</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[odor, spore-print-color, veil-color]</td>
      <td>0.995384</td>
      <td>1</td>
      <td>0.990428</td>
      <td>0.995191</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[odor, spore-print-color, stalk-color-above-ring]</td>
      <td>0.995384</td>
      <td>1</td>
      <td>0.990428</td>
      <td>0.995191</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[odor, spore-print-color, stalk-root]</td>
      <td>0.995384</td>
      <td>1</td>
      <td>0.990428</td>
      <td>0.995191</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[odor, spore-print-color, cap-surface]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[odor, spore-print-color, stalk-surface-above-...</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[odor, spore-print-color, stalk-shape]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[odor, spore-print-color, gill-color]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[odor, spore-print-color, gill-spacing]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[odor, spore-print-color, gill-attachment]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[odor, spore-print-color, bruises]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[odor, spore-print-color, ring-type]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[odor, spore-print-color, population]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[odor, spore-print-color, cap-shape]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
  </tbody>
</table>
</div>



Aaaaand we're done?! Using odor, spore-print-color and either ring number or gill size we can sucessfully find every poisonous mushroom (though we also trash a couple of edible varieties at the same time :( )

```python
X=mushroom[['odor','spore-print-color','gill-size']]
y=mushroom['class']
X=pd.get_dummies(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
dt=DecisionTreeClassifier(criterion='entropy').fit(X_train,y_train)
preds=dt.predict(X_train)
print(classification_report(y_train,preds))
preds=dt.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      0.99      0.99      3365
               p       0.99      1.00      0.99      3134
    
        accuracy                           0.99      6499
       macro avg       0.99      0.99      0.99      6499
    weighted avg       0.99      0.99      0.99      6499
    
                  precision    recall  f1-score   support
    
               e       1.00      0.99      0.99       843
               p       0.98      1.00      0.99       782
    
        accuracy                           0.99      1625
       macro avg       0.99      0.99      0.99      1625
    weighted avg       0.99      0.99      0.99      1625
    
    

```python
plt.figure(figsize=(20,20))
tree.plot_tree(dt, max_depth=None, feature_names=X_train.columns, class_names=['e','p'], label='all',filled=True, impurity=True)
plt.show()
```


    
![png](/images/mushroom_analysis_files/output_106_0.png)
    


Using these features, we miss out on 36 edible mushrooms, beccause they're on the same leaf as 37 (remember it's majority voting here!!) poisonous mushrooms. Very close call in all honesty, but a result's a result at the end of the day...

Let's go crazier!

## Challenge mode part 2 <a name="chal2"></a>

Keep going until we get a perfect score!

```python
experiments.sort_values('f1',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>[odor, spore-print-color, cap-color]</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[odor, spore-print-color, stalk-surface-below-...</td>
      <td>0.996923</td>
      <td>0.99586</td>
      <td>0.997766</td>
      <td>0.996812</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[odor, spore-print-color, stalk-color-below-ring]</td>
      <td>0.996923</td>
      <td>1</td>
      <td>0.993618</td>
      <td>0.996799</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[odor, spore-print-color, habitat]</td>
      <td>0.996769</td>
      <td>0.997761</td>
      <td>0.995533</td>
      <td>0.996646</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[odor, spore-print-color, veil-color]</td>
      <td>0.995384</td>
      <td>1</td>
      <td>0.990428</td>
      <td>0.995191</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[odor, spore-print-color, stalk-color-above-ring]</td>
      <td>0.995384</td>
      <td>1</td>
      <td>0.990428</td>
      <td>0.995191</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[odor, spore-print-color, stalk-root]</td>
      <td>0.995384</td>
      <td>1</td>
      <td>0.990428</td>
      <td>0.995191</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[odor, spore-print-color, gill-size]</td>
      <td>0.994461</td>
      <td>0.988644</td>
      <td>1</td>
      <td>0.994289</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[odor, spore-print-color, ring-number]</td>
      <td>0.994461</td>
      <td>0.988644</td>
      <td>1</td>
      <td>0.994289</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[odor, spore-print-color, gill-color]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[odor, spore-print-color, stalk-shape]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[odor, spore-print-color, cap-surface]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[odor, spore-print-color, stalk-surface-above-...</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[odor, spore-print-color, gill-spacing]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[odor, spore-print-color, gill-attachment]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[odor, spore-print-color, bruises]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[odor, spore-print-color, ring-type]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[odor, spore-print-color, population]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[odor, spore-print-color, cap-shape]</td>
      <td>0.994307</td>
      <td>1</td>
      <td>0.988194</td>
      <td>0.994062</td>
    </tr>
  </tbody>
</table>
</div>



```python
experiments=combo_finder(mushroom,['odor','spore-print-color','cap-color'],'class')
```

```python
experiments.sort_values('f1',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>[odor, spore-print-color, cap-color, habitat]</td>
      <td>0.998923</td>
      <td>0.997771</td>
      <td>1</td>
      <td>0.998884</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[odor, spore-print-color, cap-color, stalk-sur...</td>
      <td>0.998923</td>
      <td>1</td>
      <td>0.997766</td>
      <td>0.998882</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[odor, spore-print-color, cap-color, bruises]</td>
      <td>0.998307</td>
      <td>1</td>
      <td>0.99649</td>
      <td>0.998242</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[odor, spore-print-color, cap-color, stalk-root]</td>
      <td>0.998307</td>
      <td>1</td>
      <td>0.99649</td>
      <td>0.998242</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[odor, spore-print-color, cap-color, stalk-col...</td>
      <td>0.998</td>
      <td>1</td>
      <td>0.995852</td>
      <td>0.997922</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[odor, spore-print-color, cap-color, gill-size]</td>
      <td>0.997538</td>
      <td>0.994921</td>
      <td>1</td>
      <td>0.997454</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[odor, spore-print-color, cap-color, ring-number]</td>
      <td>0.997538</td>
      <td>0.994921</td>
      <td>1</td>
      <td>0.997454</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[odor, spore-print-color, cap-color, gill-atta...</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[odor, spore-print-color, cap-color, gill-spac...</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[odor, spore-print-color, cap-color, gill-color]</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[odor, spore-print-color, cap-color, stalk-shape]</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[odor, spore-print-color, cap-color, cap-surface]</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[odor, spore-print-color, cap-color, stalk-col...</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[odor, spore-print-color, cap-color, veil-color]</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[odor, spore-print-color, cap-color, ring-type]</td>
      <td>0.99723</td>
      <td>1</td>
      <td>0.994257</td>
      <td>0.99712</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[odor, spore-print-color, cap-color, population]</td>
      <td>0.997076</td>
      <td>0.993974</td>
      <td>1</td>
      <td>0.996978</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[odor, spore-print-color, cap-color, stalk-sur...</td>
      <td>0.996923</td>
      <td>0.99586</td>
      <td>0.997766</td>
      <td>0.996812</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[odor, spore-print-color, cap-color, cap-shape]</td>
      <td>0.995384</td>
      <td>1</td>
      <td>0.990428</td>
      <td>0.995191</td>
    </tr>
  </tbody>
</table>
</div>



```python
experiments=combo_finder(mushroom,['odor','spore-print-color','cap-color','habitat'],'class')
```

```python
experiments.sort_values('f1',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.999077</td>
      <td>1</td>
      <td>0.998086</td>
      <td>0.999042</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.998923</td>
      <td>0.997771</td>
      <td>1</td>
      <td>0.998884</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.998923</td>
      <td>0.997771</td>
      <td>1</td>
      <td>0.998884</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.998923</td>
      <td>0.997771</td>
      <td>1</td>
      <td>0.998884</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.998923</td>
      <td>0.997771</td>
      <td>1</td>
      <td>0.998884</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.998923</td>
      <td>0.997771</td>
      <td>1</td>
      <td>0.998884</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.998923</td>
      <td>0.997771</td>
      <td>1</td>
      <td>0.998884</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.998461</td>
      <td>0.997769</td>
      <td>0.999043</td>
      <td>0.998406</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.997692</td>
      <td>0.997766</td>
      <td>0.997447</td>
      <td>0.997607</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[odor, spore-print-color, cap-color, habitat, ...</td>
      <td>0.996615</td>
      <td>0.997442</td>
      <td>0.995533</td>
      <td>0.996487</td>
    </tr>
  </tbody>
</table>
</div>



```python
lst=[8,9,15,14,11,16,5]
for i in lst: 
    print(experiments.sort_values('f1',ascending=False).loc[i,'cols'][-1])
```

    stalk-root
    stalk-surface-above-ring
    ring-type
    ring-number
    stalk-color-above-ring
    population
    gill-size
    

So, using odor, spore print color, cap color, habitat and either stalk root, stalk surface (above ring), ring type, ring number, stalk colour (above ring), population or gill size we can perfectly classify our mushroom data. 

To be fair, this lines up very nicely with our EDA, where we saw a lot of order to the data, and a lot of features seemed to unevenly distributed across classes, as well as our random forest feature importances, where the top 10 features were types of odor, spore print colour, cap colour and habitat.

Just for fun, let's see what this final tree might look like

```python
X=mushroom[['odor','spore-print-color','cap-color','habitat','stalk-root']]
y=mushroom['class']
X=pd.get_dummies(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
```

```python
dt=DecisionTreeClassifier(criterion='entropy').fit(X_train,y_train)
preds=dt.predict(X_train)
print(classification_report(y_train,preds))
preds=dt.predict(X_test)
print(classification_report(y_test,preds))
```

                  precision    recall  f1-score   support
    
               e       1.00      1.00      1.00      3365
               p       1.00      1.00      1.00      3134
    
        accuracy                           1.00      6499
       macro avg       1.00      1.00      1.00      6499
    weighted avg       1.00      1.00      1.00      6499
    
                  precision    recall  f1-score   support
    
               e       1.00      1.00      1.00       843
               p       1.00      1.00      1.00       782
    
        accuracy                           1.00      1625
       macro avg       1.00      1.00      1.00      1625
    weighted avg       1.00      1.00      1.00      1625
    
    

```python
plt.figure(figsize=(20,20))
tree.plot_tree(dt, max_depth=None, feature_names=X_train.columns, class_names=['e','p'], label='all',filled=True, impurity=True)
plt.show()
```


    
![png](/images/mushroom_analysis_files/output_118_0.png)
    


It looks very similar to just odour!! Looks very much like as we added extra features, we gradually weeded out the 95 mushrooms that we couldn't classify based exclusively on odour. Probably more effort than its worth but I had fun, didn't you?

## B-B-B-Bonus Round (Unsupervised Clustering Exploration)<a name="c3"></a>

Kaggle says that there's 23 species of mushrooms represented in the dataset. Let's play around with some unsupervised clustering and see what happens - we won't know if we're right, but learing is its own reward :)

```python
from sklearn.cluster import KMeans,DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
```

```python
X=mushroom.copy()
X=pd.get_dummies(X,drop_first=False)
```

Since our data is all categorical, K-means is not a very meaningful algorithm to use, so my first trial is with agglomerative clustering

```python
ag = AgglomerativeClustering(n_clusters=23)
preds=ag.fit_predict(X)
preds=pd.Series(preds)
```

```python
data=pd.concat([mushroom,preds],axis=1)
```

```python
data=data.rename(columns={0:'species'})
```

We can plot the distribution of features for each cluster, to see if the clustering is meaningful. The logic here is that some features must be homogenous across a mushroom species in order for the species to be recognised as such. While some features like cap colour could conceivably vary across a species, some things must stay consistent for us to be able to identify them.

If the clustering produces an output like that, I'd be inclined to think that we might have identified at least some of the species...

```python
cols=list(data.columns)
cols.remove('species')
for col in cols:
    df_plot = data.groupby([col, 'species']).size().reset_index().pivot(columns=col, index='species', values=0)
    df_plot.plot(kind='bar',stacked=True)
```

    C:\Users\jaa31\anaconda3\lib\site-packages\pandas\plotting\_matplotlib\core.py:320: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig = self.plt.figure(figsize=self.figsize)
    


    
![png](/images/mushroom_analysis_files/output_128_1.png)
    



    
![png](/images/mushroom_analysis_files/output_128_2.png)
    



    
![png](/images/mushroom_analysis_files/output_128_3.png)
    



    
![png](/images/mushroom_analysis_files/output_128_4.png)
    



    
![png](/images/mushroom_analysis_files/output_128_5.png)
    



    
![png](/images/mushroom_analysis_files/output_128_6.png)
    



    
![png](/images/mushroom_analysis_files/output_128_7.png)
    



    
![png](/images/mushroom_analysis_files/output_128_8.png)
    



    
![png](/images/mushroom_analysis_files/output_128_9.png)
    



    
![png](/images/mushroom_analysis_files/output_128_10.png)
    



    
![png](/images/mushroom_analysis_files/output_128_11.png)
    



    
![png](/images/mushroom_analysis_files/output_128_12.png)
    



    
![png](/images/mushroom_analysis_files/output_128_13.png)
    



    
![png](/images/mushroom_analysis_files/output_128_14.png)
    



    
![png](/images/mushroom_analysis_files/output_128_15.png)
    



    
![png](/images/mushroom_analysis_files/output_128_16.png)
    



    
![png](/images/mushroom_analysis_files/output_128_17.png)
    



    
![png](/images/mushroom_analysis_files/output_128_18.png)
    



    
![png](/images/mushroom_analysis_files/output_128_19.png)
    



    
![png](/images/mushroom_analysis_files/output_128_20.png)
    



    
![png](/images/mushroom_analysis_files/output_128_21.png)
    



    
![png](/images/mushroom_analysis_files/output_128_22.png)
    


We can also look at the silhouette score for our clusters, and plot the variation of silhouette score vs cluster size

```python
silhouette=[]
for n in range(2,25):
    silag=AgglomerativeClustering(n_clusters=n)
    silpred=silag.fit_predict(X)
    silhouette.append(silhouette_score(X,silpred))    
```

```python
x=range(2,25)
y=silhouette

plt.figure()
plt.plot(x,y)
plt.title('Silhouette Score vs Cluster Size for Agglomerative Clustering')
plt.show()
a=np.argsort(silhouette)
print(f'The best silhouette score is {silhouette[a[-1]]}, achieved using {x[a[-1]]} clusters')
```


    
![png](/images/mushroom_analysis_files/output_131_0.png)
    


    The best silhouette score is 0.321556132142073, achieved using 12 clusters
    

```python
a=np.argsort(silhouette)
print(f'The best silhouette score is {silhouette[a[-1]]}, achieved using {x[a[-1]]} clusters')
```

    The best silhouette score is 0.321556132142073, achieved using 12 clusters
    

Using this model, we get the most coherent results using 12 clusters.

However, much like with our supervised model, the amount of one hot encoded data makes finding a useful distance metric hard. So what happens if we cluster using hamming distance?

Problem is, we'll have to calculate it ourselves....

```python
X_new=(2 * np.inner(X-0.5, 0.5-X) +(X.shape[1] / 2))## Got some help from StackOverflow, took me ages to get my head around it!!
                                    ## Suffice it to say it calculates the the number of bits different between each row
            
```

```python
X_new
```




    array([[ 0., 16., 20., ..., 28., 24., 26.],
           [16.,  0., 10., ..., 26., 30., 24.],
           [20., 10.,  0., ..., 24., 30., 26.],
           ...,
           [28., 26., 24., ...,  0., 30.,  6.],
           [24., 30., 30., ..., 30.,  0., 30.],
           [26., 24., 26., ...,  6., 30.,  0.]])



```python
ag1 = AgglomerativeClustering(n_clusters=23,linkage='single',affinity='precomputed')
preds1=ag1.fit_predict(X_new)
preds1=pd.Series(preds1)
```

```python
data1=pd.concat([mushroom,preds1],axis=1)
data1=data1.rename(columns={0:'species'})
```

```python
cols=list(data1.columns)
cols.remove('species')
for col in cols:
    df_plot = data1.groupby([col, 'species']).size().reset_index().pivot(columns=col, index='species', values=0)
    df_plot.plot(kind='bar',stacked=True)
```

    C:\Users\jaa31\anaconda3\lib\site-packages\pandas\plotting\_matplotlib\core.py:320: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig = self.plt.figure(figsize=self.figsize)
    


    
![png](/images/mushroom_analysis_files/output_138_1.png)
    



    
![png](/images/mushroom_analysis_files/output_138_2.png)
    



    
![png](/images/mushroom_analysis_files/output_138_3.png)
    



    
![png](/images/mushroom_analysis_files/output_138_4.png)
    



    
![png](/images/mushroom_analysis_files/output_138_5.png)
    



    
![png](/images/mushroom_analysis_files/output_138_6.png)
    



    
![png](/images/mushroom_analysis_files/output_138_7.png)
    



    
![png](/images/mushroom_analysis_files/output_138_8.png)
    



    
![png](/images/mushroom_analysis_files/output_138_9.png)
    



    
![png](/images/mushroom_analysis_files/output_138_10.png)
    



    
![png](/images/mushroom_analysis_files/output_138_11.png)
    



    
![png](/images/mushroom_analysis_files/output_138_12.png)
    



    
![png](/images/mushroom_analysis_files/output_138_13.png)
    



    
![png](/images/mushroom_analysis_files/output_138_14.png)
    



    
![png](/images/mushroom_analysis_files/output_138_15.png)
    



    
![png](/images/mushroom_analysis_files/output_138_16.png)
    



    
![png](/images/mushroom_analysis_files/output_138_17.png)
    



    
![png](/images/mushroom_analysis_files/output_138_18.png)
    



    
![png](/images/mushroom_analysis_files/output_138_19.png)
    



    
![png](/images/mushroom_analysis_files/output_138_20.png)
    



    
![png](/images/mushroom_analysis_files/output_138_21.png)
    



    
![png](/images/mushroom_analysis_files/output_138_22.png)
    


```python
silhouette=[]
for n in range(2,25):
    silag=AgglomerativeClustering(n_clusters=n,linkage='single',affinity='precomputed')
    silpred=silag.fit_predict(X_new)
    silhouette.append(silhouette_score(X_new,silpred))
    

```

```python
x=range(2,25)
y=silhouette

plt.figure()
plt.plot(x,y)
plt.title('Silhouette Score vs Cluster Size for Agglomerative Hamming Clustering')
plt.show()
a=np.argsort(silhouette)
print(f'The best silhouette score is {silhouette[a[-1]]}, achieved using {x[a[-1]]} clusters')
```


    
![png](/images/mushroom_analysis_files/output_140_0.png)
    


    The best silhouette score is 0.4483201452578324, achieved using 21 clusters
    

Interesting!! Using the hamming distance, it looks like we can split the mushrooms into clusters that are homogenous with respect to multiple features, and our max silhouette score isn't too far off the number we expected to have. That's pretty good!! However, without further information we can't be sure of our findings, but it was fun to do!!

# Summary

There not being a one-size-fits all rule for mushroom enthusiasts to identify posionous mushrooms, from this dataset we can see that there are a suprising amount of features which are strongly indicative of poison, though of course we can't be sure since the original labels weren't either!! The most prominent of these features was odor, followed by spore-print and cap colours. 

Possibly due to the synthetic nature of the dataset, or perhaps due to the strong correlation of certain features to class, K Nearest Neighbor, Decision Tree and Random Forest algorithms required little to no refinement to achieve a 100% classification rate on this data. Using just odor, this score drops only slightly to 98%. 

Using Unsupervised learning methods, it appears that ring size, ring type, stalk root, gill-size and the toxicity of a mushroom are possible heuristics to look at when deteriming mushroom species. 

That's it, thank you for humouring me! Hopefully you enjoyed this mushroomy walk as much as I did.
