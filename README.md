![](https://github.com/annah-vu/pokemon_project/blob/master/poketitle.png)
*banner generously donated by: Lupe Luna*
<br>

# Can You Catch Them All?
A Pokemon Themed Project by an Aspiring Pokemon Trainer: Anna Vu
<br>
<br>

Encountered a new Pokemon? Make sure to catch it so you can expand your Pokedex! Though handfuls of Pokemon are *pretty* easy to secure, there's a wide range of difficulty. You won't know their level of difficulty to catch unless you try, but that can get expensive quick. Pokeballs and its variants aren't free. 

<br>

If you knew how easy or hard it was to catch a Pokemon, you could take the right approaches. You don't want to waste time for the easier ones, and you want to come prepared for the harder ones. I am going to use classification machine learning algorithms to see if it can predict, on a scale, of how difficult they will be based on their features. 
<br>

Still interested? **Grab your favorite starter and let's go!** 


### Table of Contents
---

A.   [Project Overview             ](#a-project-overview)
1.   [Project Description          ](#1-project-description)
2.   [Project Deliverables         ](#2-project-deliverables)

B.  [Project Summary               ](#b-project-summary)
1.   [Goals                        ](#1-goals)
2.   [Initial Thoughts & Hypothesis](#2-hypothesis)
3.   [Findings & Next Steps        ](#3-findings--next-steps)

C. [Data Context                 ](#c-data-context)
1.   [About the Pokedex Data        ](#1-about-the-pokedex-data)
2.   [Data Dictionary              ](#2-data-dictionary)

D.  [Pipeline                     ](#d-pipeline)
1.   [Project Planning             ](#1-project-planning)
2.   [Data Acquisition             ](#2-data-acquisition)
3.   [Data Preparation             ](#3-data-preparation)
4.   [Data Exploration             ](#4-data-exploration)
5.   [Modeling & Evaluation        ](#5-modeling--evaluation)
6.   [Product Delivery             ](#6-product-delivery)

E.   [Modules                      ](#e-modules)

F.  [Project Reproduction         ](#f-project-reproduction)

<br>


<br>

### A. Project Overview
---

#### 1. Project Description

Some Pokemon are easier to catch than others, but before you try to catch one...wouldn't it be nice to know their catch rate? Hang onto your Pokeballs! In this project, I will be working with a Pokemon dataset to predict categorical catch rates of Pokemon. Catch rates of pokemon range from 3 (low catch rate, the hardest to catch) to 255 (high catch rate, the easiest). I'm going to be binning the catch rates into 5 categories:
 - 1.) 1 are the extremely hard ones
 - 2.) 2 are the hard ones
 - 3.) 3 are the mid-difficulty ones
 - 4.) 4 are the easy ones
 - 5.) 5 is the easiest

These categories are found under the simplified_catch_rate column. This allows me to be able to use classification for 5 groups. Bear in mind that Pokemon are super diverse, even within these 5 bins is a wide range of features. However, I should be able to find some relationships and patterns. 

#### 2. Project Deliverables

- GitHub repository and this README with project overview, goals, findings, conclusion and summary
- Jupyter Notebook with a complete walkthrough of the data science pipeline, and commented with takeaways
- Any Python module(s) used to automate processes in the project. 
<br>
<br>

### B. Project Summary
---

#### 1. Goals

My goal is to predict categorical catch rates of Pokemon based on some of their features. My classification model should be able to beat the baseline (assuming every Pokemon has the same mode catch rate category.) I'm hoping to find some meaningful relationships between the features that determine catch rate. 

#### 2. Hypothesis

I believe a number of things determine the general catchability of a Pokemon. Putting level aside, I believe that type, battle statistics, kinds of abilities, and size would determine if a Pokemon would be easier or harder to catch. 

#### 3. Findings & Next Steps

With some exploration with how some of the features correlated to each other (plus a failed attempt to find meaningful clusters), and SelectKBest feature engineering, I was able to find that total points was the biggest driver to the difficulty level of catching a Pokemon. Total Points seems to be directly related to stats like attack, defense, speed, and health points. This was a strong factor to determine catchability, but it was not the end-all. There were Pokemon with top-tier total points that actually had a very high catch rate.

The next phase is to further create new features from the original data, and perhaps separate the different catch rates further more. I believe my model would perform better with a larger scale of catch rates. Due to the constraint of time (and maybe my lack of knowledge of Pokemon), I categorized them based on how I felt was appropriate, but to categorize such a wide range of catch rates may have affected my model's ability to accurately predict which subgroup they belonged in (especially the easier ones). 
<br>

My decision tree model performed on unseen data with 71.29% accuracy. It did really well at predicting catch rates for those with either a 1 or 2 (the difficult) for their catchability, but with the two easiest subgroups---it had a really hard time distinguishing. I guess we better take this to Professor Oak's lab and look further into those easier Pokemon to see if we should categorize them in the same level of difficulty, or find features that will distinguish them.

<br>
<br>

### C. Data Context
---

#### 1. About the Pokedex Data

This Pokedex.csv was acquired from [Mario Tormo's Complete Pokemon Dataset](https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420) (at the time, it was the csv that was updated May 2020). It has over a thousand entries of Generation 1-8 Pokemon, complete with their names, stats, breeding, gender, types, what kinds of attacks they are affected by or not affected by, and more! I took some of the columns and enumerated them so that they could be used by the model. None of the the new features I made from object columns were strongly correlated, so I didn't end up using them for this goal. However, I'm sure whenever I can make other predictions, I will put them to use. 


#### 2. Data Dictionary

The Pokedex.csv reference guide, if you will. 

| Column Name             | Description                                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------------|
| pokedex_number          | Pokédex number for that specific Pokemon                                                        |
| name                    | Pokemon's name                                                                                  |
| generation              | The generation this Pokemon was introduced                                                      |
| status                  | Pokemon's status type (normal, sub legendary, legendary, mythical)                              |
| species                 | Pokemon's categorical species                                                                   |
| type_number             | Number of types that Pokemon has                                                                |
| height_m                | Pokemon's height in meters                                                                      |
| weight_kg               | Pokemon's weight in kilograms                                                                   |
| abilities_number        | Number of abilities the Pokemon has                                                             |
| ability_1               | first ability                                                                                   |
| ability_2               | second ability (none if N/A)                                                                    |
| ability_hidden          | hidden ability of the Pokemon (none if N/A)                                                     |
| hp                      | health points                                                                                   |
| attack                  | attack stat                                                                                     |
| defense                 | defense stat                                                                                    |
| sp_attack               | special attack stat                                                                             |
| sp_defense              | special defense stat                                                                            |
| speed                   | speed of the Pokemon                                                                            |
| catch_rate              | Pokemon's catch rate                                                                            |
| base_friendship         | base friendship when caught/acquired                                                            |
| base_experience         | base experience when caught/acquired                                                            |
| growth_rate             | growth rate of the Pokemon                                                                      |
| egg_type_number         | how many egg types the Pokemon has                                                              |
| egg_type_1              | egg type                                                                                        |
| egg_type_2              | 2nd egg type (none if N/A)                                                                      |
| percentage_male         | percentage of male in species of Pokemon                                                        |
| egg_cycles              | number of cycles the egg has                                                                    |
| against_?               | Eighteen features that denote the amount of damage taken against an attack of a particular type |
| is_genderless           | denotes if species of Pokemon is genderless                                                     |
| ???_num                 | categorical column that is encoded for ML                                                       |
| simplified_catch_rate * | simplified catch rate from 1-5. 1 being the hardest, 5 being the easiest.                       |





&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  * Target variable

<br>
<br>

### D. Pipeline
---

#### 1. Project Planning
:books: **Plan** ➜ ☐ _Acquire_ ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build this README containing:
    - Project overview
    - Initial thoughts and hypotheses
    - Data Dictionary
    - Walkthrough of the data science pipeline
    - Project summary
    - Instructions to reproduce
- [x] Plan stages of project and consider needs versus desires
    - Think about what target to do
    - Make a Trello Board (https://trello.com/b/pk8AN3Ru/individual-project)
    - Refresh on how to carry the tasks out

#### 2. Data Acquisition
✓ _Plan_ ➜ :open_book: **Acquire** ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Find a dataset about Pokemon
- [x] Observe data structure
- [x] Save it to a local .csv for use. 

#### 3. Data Preparation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ :soap: **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Address missing values, and outliers. Assure all values are reasonable. 
- [x] Make any desirable object columns into machine-learning-friendly columns.
- [x] Create new features
- [x] Split data into train, validate, and test sets. 
- [x] Make my target variable by binning the catch rates 


#### 4. Data Exploration
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ :mag: **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Explore univariate data
- [x] Explore relationships between variables between each other, and the target.
- [x] Form hypothesis and run statistical testing
- [x] Feature engineering with built in scikit modules

#### 5. Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ :dart: **Model** ➜ ☐ _Deliver_

- [x] Establish baseline prediction
- [x] Create, fit, and predict with models
- [x] Evaluate models with out-of-sample data
- [x] Run best performing model on test data, and evaluate. 

#### 6. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ :white_check_mark: **Deliver**
- [x] Prepare Jupyter Notebook with thorough walk-through of the data science pipeline
- [x] Address next steps

<br>
<br>

### E. Modules
---

 - wrangle.py = contains acquire and prepare functions used to retrieve and prepare the Pokedex for use.
 - explore.py = contains functions I used to explore, visualize, and run statistical tests.

<br>
<br>

### F. Project Reproduction
---

Should you want to be a Pokemon master too, you can recreate this project with some simple steps. 
 - Download the csv from the Kaggle link [here](https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420)
 - Download helper function files
 - Download final_catch_rates.ipynb notebook
 - Become a champion.

<br>

Good luck on your journey, I will be doing more research! 
See you around!

