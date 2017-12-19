#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:14:00 2017
@author: kahmunleong
Assignment: Answers of Assessment for Global Delivery Pod 
"""
############################## Preliminaries ##################################
# Modules used:
from numpy import*
from urllib.request import*
from urllib.error import*
from pandas import*
from collections import*
import shutil
import csv
import requests
import nltk
# nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import ascii_lowercase
import os, re, itertools, nltk, snowballstemmer
pattern.download('all')
import inflect 
import multiprocessing
from multiprocessing import Pool

################################### Q1 ########################################
#------------------------------------------------------------------------------
# Write a function in python to sum up a given set of numbers other than itself
#------------------------------------------------------------------------------
def sum_neighbours(nums):
   # Aim: To calculate sum of neighbours for number[i] from given list.
   # Input: A list of numbers(num)
   # Output: A list of the sum of neighbours for number[i] from given list.
    output = []
    for i in range(len(nums)):
        left = nums[:i][-len(nums):] # all numbers on the left of nums[i]
        right= nums[i+1:len(nums)+i+1] # all numbers on the right of nums[i]
        new = left + right 
        # Creates n list of neighbour numbers for n numbers in given list
        output.append(sum(new))  
        # sum all elements in each list and append it to single list
    return(output)
    
################################### Q2 ########################################
#------------------------------------------------------------------------------
# a) Write code to download Kaggle dataset
#------------------------------------------------------------------------------
# Code to download Kaggle data set: 
def kaggle_download(kaggle_url, local_filename):
    # Aim: To download code from kaggle url
    # Input: Url for dataset(kaggle_url),
    #        Name of local file( local_filename)
    # Output: Local csv file
    response = urlopen(kaggle_url)
    d = response.read()
    with open(local_filename, 'wb') as f: 
            f.write(d)

# Raw dataset url loaded in safari browser:
q2_url = 'https://storage.googleapis.com/kaggle-datasets/2169/3651/Sales_Transactions_Dataset_Weekly.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1513651992&Signature=TQW3ayyq7fM9PwWQzn5zJyRqfKenydOhE482EK7UMTuiwteLd0ujwi4mYx6mHa0PLWQmOEVxZ%2FhADDrI%2FrxB51ZEHV9br3oBOCtLrdHoJOe3uD%2B3jF93oQrtLclV9L3KNYrF7RC44H737mAuE1I7EcYpLYNiJYnuP97qOuKwGZXyFT1kqIDQ3fgT%2FT%2BjftWB9Thh3uRVq4%2BFeWloibwT%2FgqTQTj5fR46wfpwmkzx1dwRmZNiMM%2FPSGEGCykhtRroHTgwEJp4kcTzAiQGDlJ5iJLKziUvf%2FOQM4C%2F6BaJdELlIvJOtMp9JcwlniYrKNnWzF%2BrUZ2CbyCwf2CQCxuF%2FQ%3D%3D'
# Name for local file:
q2_filename = 'salesdata.csv'
# Use function to download code from kaggle to local file         
kaggle_download(q2_url, q2_filename) 

"""        
Notes:
1)  The above method requires login to Kaggle account online.
    Alternatively, one could both login and download data as follows:    
    # Kaggle Username and Password
    kaggle_info = {'UserName': config.kaggle_username,
               'Password': config.kaggle_password}
    r = requests.get(q2_url)
    # Login to Kaggle and retrieve the data.
    r = requests.post(r.url, data = kaggle_info, stream=False)
    # Writes the data to a local file one chunk at a time.
    f = open(q2_filename,"wb")
    for chunk in r.iter_content(chunk_size = 512 * 1024): 
        # Reads 512KB at a time into memory
        if chunk: # filter out keep-alive new chunks
            f.write(chunk)
        f.close()

2)  Can also use dataset without downloading to local file (need online login):
    sales_data = read_csv(q2_url)
"""
#------------------------------------------------------------------------------
# Load data from local file to answer the rest of Q2:
sales_data = read_csv(q2_filename)

#------------------------------------------------------------------------------
# b) Identify the best performing product (based on volume)
#------------------------------------------------------------------------------
# Data frame containing only volume sold accross 52 weeks:
sales_vol = sales_data.iloc[:, 1:53] 

# Get total volume sold across 52 weeks for each product:
vol = sales_vol.sum(axis=1)

# Add total volume sold column to sales_data data frame:
sales_data['total volume'] = vol

# Returns the row corresponding to product with highest volume sold:
best_product = sales_data.loc[sales_data['total volume'].idxmax()]
 
 
# Answer: The best performing product (highest volume sold) is the product        
# ~~~~~~~ with product code P409 (total volume sold = 2220 accross 52 weeks).      
     
#------------------------------------------------------------------------------
# c) Identify the most promising product (emerging product)
#------------------------------------------------------------------------------
# To find the most promising product(emerging product), I compute the sales
# growth (%) by comparing (arbritrarily chosen) last four weeks to the first 
# four: ((volume first four - volume last four)/volume first four)*100

# Compute total volume sold for first four weeks and last four weeks:
first_four_weeks = sales_vol.iloc[:,0:4]
last_four_weeks = sales_vol.iloc[:,48:]
prior = first_four_weeks.sum(axis = 1)
current = last_four_weeks.sum(axis = 1)

# Compute sales growth (%): 
sales_growth = ((current - prior)/prior)*100

# Add sales growth column to sales data set:
sales_data['sales growth'] = sales_growth

# Returns the row corresponding to product with highest sales growth (%):
emerging_product = sales_data.loc[sales_data['sales growth'].idxmax()]

# Answer: The emerging product (highest sales growth) is product with       
# ~~~~~~~ product code P816 (sales growth of 2000 % in last four weeks 
#         from first four weeks).      

#------------------------------------------------------------------------------
# d) Identify the worst performing product on a biweekly basis
#------------------------------------------------------------------------------
# To find the worst performing product, I again use the sales growth rate(%):
# (current volume - prior volume)/(prior volume*100) 
# This time sales growth is computed biweekly.

# volume from even weeks (current volume):
even_weeks = sales_vol.iloc[:, ::2]
# volume from odd weeks (prior volume):
odd_weeks = sales_vol.iloc[:, 1::2]

# Total volume for every two weeks: 
biweekly = even_weeks + odd_weeks.values
prior_biweekly = biweekly_sum.iloc[:, ::2]
current_biweekly = biweekly_sum.iloc[:, 1::2]

# Calculate biweekly sales growth rate(%):
biweekly_growth = ((current_biweekly - prior_biweekly.values)/prior_biweekly.values)*100

biweekly_growth.min(axis = 1) # The minimum growth rate for each products 
                              # across 52 weeks is -100%. There are ties for 
                              # some products so...                         
# count the number of times -100% occured for each product.
biweekly_min_count = biweekly_growth.isin({-100}).sum(1)
 
# Add minimum biweekly sales growth count to sales data set:                           
sales_data['minimum biweekly sales growth count'] = biweekly_min_count 

# Returns the row corresponding to product with highest count for lowest 
# biweekly sales growth (%):
worst_product = sales_data.loc[sales_data['minimum biweekly sales growth count'].idxmax()]

# Answer:  The worst performing product (-100% growth rate occurs the most)  
# ~~~~~~~  on a biweekly basis is the product with product code P349 
#          (-100% sales growth rate occured 7 times on a biweekly basis).

#------------------------------------------------------------------------------
# e) Identify outliers from the data and output corresponding week numbers
#------------------------------------------------------------------------------
# I am going to find outliers for total products sold in each week
# based on interquartile range (IQR).

# Get total volume of product sold for each week:
weekly_sales = sales_vol.sum()

# Compute first quarter, second quartile and IQR:
q1 = weekly_sales.quantile(0.25)
q3 = weekly_sales.quantile(0.75)
iqr = q3 - q1

# Get the bounds to determine outliers:
upper_bound = q3 + 1.5*iqr
lower_bound = q1 - 1.5*iqr

# Check if data is within bounds:
weekly_sales > upper_bound
weekly_sales < lower_bound

# Answer: The sales are within bounds so there are no outliers for total 
# ~~~~~~~ products sold in each week.

################################### Q3 ########################################
#------------------------------------------------------------------------------
# a) Reuse code from Q2 to download Kaggle dataset:
#------------------------------------------------------------------------------
# same as Q2:
q3_url = 'https://www.kaggle.com/madhab/jobposts/downloads/data%20job%20posts.csv'
q3_filename = 'datajobposts.csv'
kaggle_download(q3_url, q3_filename) 

#------------------------------------------------------------------------------
# Load data from local file to answer the rest of Q3
jobs_data = read_csv(q3_filename)
#------------------------------------------------------------------------------
# b) Extract the following fields from the jobpost column 
#------------------------------------------------------------------------------
# Prep data for extraction:
ori_list = ["\r", "\n", "TITLE:",
          "POSITION DURATION:", "DURATION:",
          "POSITION LOCATION:", "LOCATION:", 
          "JOB DESCRIPTION:", "DESCRIPTION:", 
          "JOB RESPONSIBILITIES:", "RESPONSIBILITIES:",
          "REQUIRED QUALIFICATIONS:", 
          "REMUNERATION/ SALARY:", "REMUNERATION:", 
          "APPLICATION DEADLINE:",
          "ABOUT COMPANY:"]

modify_list = [" ", " ", "% TITLE:",
     "DURATION:", "% DURATION:", 
     "LOCATION:", "% LOCATION:", 
     "DESCRIPTION:", "% DESCRIPTION:",
     "RESPONSIBILITIES:","% RESPONSIBILITIES:",
     "% REQUIRED QUALIFICATIONS:",
     "REMUNERATION:", "% REMUNERATION:", 
     "% APPLICATION DEADLINE:", 
     "% ABOUT COMPANY:"]

d1 = jobs_data['jobpost'].replace(ori_list , value=modify_list, regex=True) 

# Extract: 
# 1. Job Title
job_title = d1.str.extract('TITLE:(.*?)%', expand=False).str.strip()
# 2. Position Duration
position_duration = d1.str.extract('DURATION:(.*?)%', expand=False).str.strip()
# 3. Position Location
position_location = d1.str.extract('LOCATION:(.*?)%', expand=False).str.strip()
# 4. Job Description
job_description = d1.str.extract('DESCRIPTION:(.*?)%', expand=False).str.strip()
# 5. Job Responsibilities
job_responsibilities = d1.str.extract('RESPONSIBILITIES:(.*?)%', expand=False).str.strip()
# 6. Required Qualifications
required_qualifications = d1.str.extract('REQUIRED QUALIFICATIONS:(.*?)%', expand=False).str.strip()
# 7. Remuneration
remuneration = d1.str.extract('REMUNERATION:(.*?)%', expand=False).str.strip()
# 8. Application Deadline
p1 = r'(\d\d\s*[ ,./-]*(?:January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*[ ,./-]*\d{4})'
application_deadline = d1.str.extract(p1, expand=False).str.strip()
# 9. About Company
about_company = d1.str.extract('ABOUT COMPANY:(.*)', expand=False).str.strip()

#------------------------------------------------------------------------------
# c) Identify the company with the most number of job ads in the past 2 years
#------------------------------------------------------------------------------
# Check last two years:
unique(jobs_data['Year']) # Last two years is 2014 and 2015

# subset the jobs data set wrt company names that listed jobs 
# in year 2014 and 2015.
company_twoyears = jobs_data.loc[jobs_data['Year'] > 2013, 'Company'] 

# Count the number of times a company name appears in the data set for past two 
# years arranged from most times to least:
Counter(company_twoyears).most_common()[0]

# Answer:   The company with the most number of job ads in the past two years 
# ~~~~~~~   (company name appearing most in 2014 - 2015) is 
#           Mentor Graphics Development Services CJSC with 83 ads.

#------------------------------------------------------------------------------
# d) Identify the month with the largest number of job ads over the years
#------------------------------------------------------------------------------
# Months column of job dataset:
months = jobs_data['Month']

# Count the number of timse a month appears in the data set arranged from most 
# times to least:
Counter(months).most_common()[0]

# Answer:   The month with the largest number of job ads over the years is  
# ~~~~~~~   March with a total of 1702 ads.

#------------------------------------------------------------------------------
# e) Find median, mean, min and max values for each product
#------------------------------------------------------------------------------
#**** Didn't answer question because not sure what is product referring to ****

#------------------------------------------------------------------------------
# f) Clean text and generate new text from Job Responsibilities column: 
#   new text shall not contain any stop words, and the plural words shall be 
#   converted into singular words.
#------------------------------------------------------------------------------
# stop words from nltk package:
stopset = stopwords.words('english')

# Convert floats to strings:
no_float = job_responsibilities.replace(nan, 'nan', regex=True) 

# Job Responsibilities column without stop words:
no_stop_words = no_float.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopset)]))


# Split words into a series of list of strings from no stop words column series:
wnl = nltk.WordNetLemmatizer()
s = no_stop_words.apply(lambda x:[wnl.lemmatize(word) for word in nltk.wordpunct_tokenize(x)])

# Extract the plural words from the list:
p = inflect.engine()
pl = s.apply(lambda x: [word for word in x if p.singular_noun(word)])

# Store all plurals from p1 into one list:
pl_list = list(itertools.chain.from_iterable(pl))  

# Find the singular version of the plurals:
sin_list = [p.singular_noun(plural) for plural in pl_list]

# Replace the plurals with the singulars:
c = no_stop_words.replace(pl_list, value=sin_list, regex=True) 

# c is the cleaned job responsibilities column with no stop words and 
# plural words are converted to singular words.
#------------------------------------------------------------------------------
# g) Store the results in a new Dataframe/SQL table
#------------------------------------------------------------------------------

# Define new data frame column title and values:
h = {'Job title': job_title, 'Position duration': position_duration , 
     'Position location': position_location, 
     'Job Description': job_description ,
     'Job responsibilities': c,
     'Required qualifications': required_qualifications,
     'Renumeration':remuneration, 'Application deadline': application_deadline,
     'About Company':about_company }

# new data frame: 
new_df = DataFrame(data=h)
################################### Q4 ########################################
#------------------------------------------------------------------------------
# a) Download test.csv from
#------------------------------------------------------------------------------
# Downloaded only the test.csv file from link provided

#------------------------------------------------------------------------------
# b) Load data to Pandas data frame from local path.
#------------------------------------------------------------------------------
test = read_csv("test.csv")
 
#------------------------------------------------------------------------------
# c) Calculate similarity between description_x and description_y and 
#    store resultant scores in a new column
#------------------------------------------------------------------------------
# Since there is no specified method in the question for the calculation 
# of similarity of the two columns, I decided to use cosine similarity. 
# To fill in the same_security column, one can then easily use the cosine 
# similarity together with somr arbritrary threshold (say: 0.4) to determine
# if the security in description_x and description_y are the same.

# Define Cosine similarity function: 
def cosine_sim_vec(vec1, vec2):
    # Aim: Compute cosine similarity
    # Input: two lists containing words and counts
    # Output: a number ranging from 0 to 1 
    #         ( 0 = no similarity and 1 = exact same) 
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

# Define cosine similarity function for data frame:
def cosine_sim_df(df):
    # Aim: To compute cosine similarity for data frame
    similarity = []
    # split the words in the string by space in each row for both columns:
    split_x = test["description_x"].str.lower().str.split()
    split_y = test["description_y"].str.lower().str.split()
    # Count the occurence of each word: 
    s1 = split_x.apply(Counter)
    s2 = split_y.apply(Counter)
    for i in range(len(test)):
        g = cosine_sim_vec(s1[i], s2[i])   
        similarity.append(g)
    return(similarity)

# Compute cosine similarity:
similarity = cosine_sim_df(test)

# Store results in new column of data
test["similarity"] = similarity 

#------------------------------------------------------------------------------
# d) Parallelise the matching process (bonus)
#------------------------------------------------------------------------------
pool = multiprocessing.Pool()

num_partitions = 10 #number of partitions to split dataframe
num_cores = 4

# Define function to parallelise data frame:
def parallelise_dataframe(df, func):
    df_split = array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pool.map(func, df_split)
    pool.close()
    pool.join()
    return df

# Execute:
res = parallelize_dataframe(test, cosine_sim_df)
