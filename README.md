# Can You Catch Them All?
A Pokemon Themed Project by: Anna Vu

I played Pokemon when I was really young, and I recently started getting back into it. I wonder if my data science skills can help me know more about the game before I actually play it. 



### Table of Contents
---

I.   [Project Overview             ](#i-project-overview)
1.   [Description                  ](#1-description)
2.   [Deliverables                 ](#2-deliverables)

II.  [Project Summary              ](#ii-project-summary)
1.   [Goals                        ](#1-goals)
2.   [Initial Thoughts & Hypothesis](#2-initial-thoughts--hypothesis)
3.   [Findings & Next Phase        ](#3-findings--next-phase)

III. [Data Context                 ](#iii-data-context)
1.   [Database Relationships       ](#1-database-relationships)
2.   [Data Dictionary              ](#2-data-dictionary)

IV.  [Process                      ](#iv-process)
1.   [Project Planning             ](#1-project-planning)
2.   [Data Acquisition             ](#2-data-acquisition)
3.   [Data Preparation             ](#3-data-preparation)
4.   [Data Exploration             ](#4-data-exploration)
5.   [Modeling & Evaluation        ](#5-modeling--evaluation)
6.   [Product Delivery             ](#6-product-delivery)

V.   [Modules                      ](#v-modules)

VI.  [Project Reproduction         ](#vi-project-reproduction)

<br>


<br>

### I. Project Overview
---

#### 1. Description

Some Pokemon are easier to catch than others, but before you try to catch one...wouldn't it be nice to know their catch rate? Hang onto your Pokeballs! In this project, I will be working with a Pokemon dataset to predict categorical catch rates of Pokemon. Catch rates of pokemon range from 3 (the hardest to catch) to 255 (the easiest). I'm going to be binning the catch rates into 5 categories:
 - 1.) 1 are the extremely hard ones
 - 2.) 2 are the hard ones
 - 3.) 3 are the mid-difficulty ones
 - 4.) 4 are the easy ones
 - 5.) 5 is the easiest

These categories are found under the simplified_catch_rate column. This allows me to be able to use classification for 5 groups. Bear in mind that Pokemon are super diverse, even within these 5 bins is a wide range of features. However, I should be able to find some relationships and patterns. 

#### 2. Deliverables

- GitHub repository and this README with project overview, goals, findings, conclusion and summary
- Jupyter Notebook with a complete walkthrough
- Any Python module(s) used to automate processes


### II. Project Summary
---

#### 1. Goals

My goal is to predict categorical catch rates of Pokemon based on some of their features. My classification model should be able to beat the baseline (assuming every Pokemon has the same mode catch rate.) I'm hoping to find some meaningful relationships between the features that determine catch rate. 

#### 2. Initial Thoughts & Hypothesis

I believe a number of things determine the general catchability of a Pokemon. Putting level aside, I thought type, kinds of abilities, and size would determine if a Pokemon would be easier or harder to catch. 

#### 3. Findings & Next Phase

With some exploration with how some of the features correlated to each other (a bit of clustering), and SelectKBest feature engineering, I was able to find that total points was the biggest driver to the difficulty level of catching a Pokemon. Total Points seems to be directly related to stats like attack, defense, speed, and health points. The next phase is to further create new features from the original data, and perhaps separate the different catch rates further more. I believe my model would perform better with a larger scale of catch rates. Due to the constraint of time (and maybe my lack of knowledge of Pokemon), I categorized them based on how I felt was appropriate, but to categorize such a wide range of catch rates may have affected my model's ability to accurately predict which subgroup they belonged in. 

### III. Data Context
---

#### 1. About the Pokedex.csv

This Pokedex.csv was acquired from https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420 (at the time, it was the csv that was updated May 2020). It has over a thousand entries of Generation 1-8 Pokemon, complete with their names, stats, breeding, gender, types, what kinds of attacks they are affected by or not affected by, and more! I took some of the columns and enumerated them so that they could be used by the model. None of the the new features I made from object columns were strongly correlated, so I didn't end up using them for this goal. However, I'm sure whenever I can make other predictions, I will put them to use. 


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
| simplified_catch_rate * | simplified catch rate from 1-5                                                                  |





&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  * Target variable

### IV. Process
---

#### 1. Project Planning
🟢 **Plan** ➜ ☐ _Acquire_ ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build this README containing:
    - Project overview
    - Initial thoughts and hypotheses
    - Project summary
    - Instructions to reproduce
- [x] Plan stages of project and consider needs versus desires
    - Think about what target to do
    - Make a Trello Board (https://trello.com/b/pk8AN3Ru/individual-project)
    - Refresh on how to carry the tasks out

#### 2. Data Acquisition
✓ _Plan_ ➜ 🟢 **Acquire** ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Find a dataset about Pokemon
- [x] Observe data structure
- [x] Save it to a local .csv for use. 

#### 3. Data Preparation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ 🟢 **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Address missing values, and outliers. Assure all values are reasonable. 
- [x] Make any desirable object columns into machine-learning-friendly columns.
- [x] Create new features
- [x] Split data into train, validate, and test sets. 
- [x] Make my target variable by binning the catch rates 


#### 4. Data Exploration
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ 🟢 **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Explore univariate data
- [x] Explore relationships between variables
- [x] Form hypothesis and run statistical testing
- [x] Feature engineering

#### 5. Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ 🟢 **Model** ➜ ☐ _Deliver_

- [x] Establish baseline prediction
- [x] Create, fit, and predict with models
- [x] Evaluate models with out-of-sample data
- [x] Run best performing model on test data, and evaluate. 

#### 6. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ 🟢 **Deliver**
- [x] Prepare Jupyter Notebook with thorough walk-through of the data science pipeline
- [x] Address next steps

### V. Modules
---

 - wrangle.py = contains acquire and prepare functions. I filled in nulls to what I found to be reasonable by observing the pokedex.csv. 
 - explore.py = contains functions I used to explore



### VI. Project Reproduction
---

Should you want to be a Pokemon master too, you can recreate this project with some simple steps. 
 - Download the csv from the Kaggle link here: https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420
 - Download helper function files
 - Download notebook
 - Become a champion.

[[Return to Top]](#can-you-catch-them-all?)
