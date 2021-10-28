''' Import modules for reidentification attack'''
import pandas as pd
import numpy as np
import random
import requests
import string
import uuid
import time
from faker import Faker
from datetime import datetime
import scipy.stats as ss
import matplotlib.pyplot as plt
import zipcodes as zc
from tqdm import tqdm
import logging

diseases = {
    9:"High Blood Pressure",
    8:"Alzheimer",
    7:"Heart Disease",
    6:"Depression",
    5:"Arthritis",
    4:"Osteoporosis",
    3:"Diabetes",
    2:"COPD",
    1:"Cancer",
    0:"Stroke"
}

disease_numbers = {
    9: 0,
    8: -1,
    7: 1,
    6: -2,
    5: 2,
    4: -3,
    3: 3,
    2: -4,
    1: 4,
    0: -5
}

def do_encode(df, cols, diseases):
    """ Encodes variables to be compatible with smartnoise
    Args:
        df = df
        cols = columns to be encoded
        diseases = dictionary of potential diseases
    Returns:
        df_enc = new data frame with encoded variables
    """
    df_enc = df.copy()
    for _ in cols:
        if _ == "Diagnosis":
            df_enc[f'{_}_encoded'] = df_enc[_].map({v: k for k, v in diseases.items()})
            df_enc[f'{_}_encoded'] = df_enc[f'{_}_encoded'].astype(int)
        elif _ == "Gender":
            df_enc[f'{_}_encoded'] = df_enc[_].replace({"F": 0, "M": 1})
            df_enc[f'{_}_encoded'] = df_enc[f'{_}_encoded'].astype(int)
        else:
            df_enc[f'{_}_encoded'] = df_enc[_].astype(int)
    return df_enc[[f'{_}_encoded' for _ in cols]]

def get_medical_data(n, lang, disease_numbers, k, logger):
    """ Create medical data set
    Args:
        n = amount of data to be created (non-anonymized)
        lang = language for personal data, such as names
        k = level of anonymization
        logger = pass a custom logger
    Returns:
        df = returns medical data set
    """
    custodian_id = []
    age = []
    gender = []
    zipcode = []
    diagnosis = []
    treatment = []
    severity = []
    duplicate_test = 0
    fake = Faker(lang)
    logging.info('Generating demographic examples')
    for n in tqdm(range(n)):
        valid=False
        while valid == False:
            # Generate values and append it to lists
            gender_select = random.choice(["M", "F"])
            age_select = random.choice(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89'])
            if lang == "de-DE":
                zipcode_select = f"{random.choice([_ for _ in list(zips['zipcode']) if len(str(_)) > 4])[0:3]}**"
            else:
                zipcode_select = f"{fake.address()[-5:][0:3]}**"
            df_temp = pd.DataFrame([gender, age, zipcode]).transpose()
            if len(df_temp[(df_temp[0]==gender_select) & (df_temp[1]==age_select) & (df_temp[2]==zipcode_select)]) > 0:
                duplicate_test += 1
                continue
            else:
                valid=True
                gender.append(gender_select)
                zipcode.append(zipcode_select)
                age.append(age_select)
            custodian_id = [uuid.uuid4().hex for i in range(len(gender) * k)]
            treatment = [f'0{str(random.randint(20,50))}' for i in range(len(gender) * k)]
            severity = [random.choice(['recovered', 'unchanged', 'intensive care']) for i in range(len(gender) * k)]
    if k > 0:
        gender = [item for item in gender for i in range(k)]
        age = [item for item in age for i in range(k)]
        zipcode = [item for item in zipcode for i in range(k)]
    diagnosis = assign_ndis(len(zipcode), diseases, disease_numbers, True) 
    df = pd.DataFrame([custodian_id, gender, age, zipcode, diagnosis, treatment, severity]).transpose()
    df.columns = ["ID", "Gender", "Age", "Zip", "Diagnosis", "Treatment", "Outcome"]
    logging.info(f'Finished generating demographic examples, had to mitigate {duplicate_test} duplicate tests.')
    return df

def assign_ndis(n, diseases, disease_numbers, transform_disease):
    """ Assign diseases with normal distribution, as it's not realistic that they have the same share
    Args:
        n = amount of data to be generated
        diseases = dictionary of diseases
    Returns:
        disease_list = list of diseases in normal distribution
    """
    # Simulate normal distribution
    x = np.arange(-5, 5)
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size = n, p = prob)
    # Substitute values with diagnoses
    disease_list = [{v:k for k, v in disease_numbers.items()}.get(item, item) for item in list(nums)]
    if transform_disease:
        disease_list = pd.DataFrame({'diseases': disease_list, 'diseases_code': disease_list})['diseases'].map(diseases)
    return list(disease_list)

def get_demographic_information(df, lang, logger):
    """ Create demographic data set based on medical data set
    Args:
        df = medical data set
        lang = language for personal data, such as names
        logger = pass a custom logger
    Returns:
        df_dem = demographic data set"""
    custodian_id = []
    name = []
    gender = []
    age = []
    zipcode = []
    fake = Faker(lang)
    logging.info(f'Create demographic data set based on medical data.')
    # Iterate through data set and set demographic information
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        profile = fake.simple_profile()
        while profile['sex'] != row['Gender']:
            profile = fake.simple_profile()
        name.append(profile['name'])
        gender.append(profile['sex'])
        age.append(random.randint(int(row['Age'][:2]), int(row['Age'][3:])))
        zipcode.append(f'{row["Zip"][:3]}{random.randint(10, 99)}')
        #zipcode.append(zc.similar_to(row['Zip'][:3])[random.randint(0, len(zc.similar_to(row['Zip'][:3])) - 1)]['zip_code'])
    logging.info('finished k-anonymization')
    df_dem = pd.DataFrame([list(df['ID']), name, gender, age, zipcode]).transpose()
    df_dem.columns = ['ID', 'Name', 'Gender', 'Age', 'Zip']
    logging.info(f'Returning dataset with length of {len(df_dem)}')
    return df_dem

def create_histogram(df, df2, col, diseases):
    """ Create histogram to compare distribution of variable in comparison between non-synthesized and synthesized data
    Args:
        df = raw, non-synthesized data
        df2 = synthesized data
        diseases = dictionary of diseases to assign it to the plot
    Returns:
        None, will render and display the graph in the notebook
    """
    fig, ax = plt.subplots(figsize=(15,8))
    ax = plt.style.use('seaborn-deep')
    ax = plt.hist([df['Diagnosis_encoded'].map(diseases), df2['Diagnosis_encoded'].map(diseases)], label=['Original', 'Synthetic'])
    ax = plt.legend(loc='upper right')
    ax = plt.xlabel('Values of Diagnosis')
    ax = plt.ylabel('Value Counts')
    fig.autofmt_xdate()
    plt.show()

def try_reidentification(df_demographic, df_medical, logger):
    """ Try patient reidentification
    Args:
        df_demographic = demographic data set generated above 
        df_medical = medical data set generated above
        logger = pass a custom logger
    Returns:
        df_dem = demographic data set
    """
    df_reident = pd.DataFrame(columns=df_medical.columns)
    logging.info(f'Performing reidentification with anonymized data: {len(df_medical)}. Attacker collection: {len(df_demographic)}')
    expand = df_medical.Age.str.split("-", n = 1, expand = True)
    df_medical['Age_Low'] = expand[0].astype(int)
    df_medical['Age_High'] = expand[1].astype(int)
    # Create data
    for index, row in tqdm(df_demographic.iterrows(), total=df_demographic.shape[0]):
        df_filtered = df_medical.loc[(df_medical.Gender == row['Gender']) & (df_medical['Age_Low'] <= row['Age']) & (df_medical['Age_High'] >= row['Age']) & (df_medical['Zip'].str.startswith(row['Zip'][:3]))].copy()
        #logging.warning(len(df_filtered))
        if len(df_filtered) == 0:
            continue
        df_filtered = df_filtered.sample(1)
        df_filtered['Name'] = row['Name']
        df_filtered['Age'] = row['Age']
        df_filtered['Zip'] = row['Zip']
        df_filtered['ID_Compare'] = row['ID']
        df_reident = df_reident.append(df_filtered)
    # Return information about potential matches
    logging.info(f'Identified {len(df_reident)} potential matches!')
    # logging.info(f'The number of potential matches might be higher than the actual data set, as a row can match to multiple published rows!')
    # Validate
    logging.info(f'Validating IDs ...')
    res = []
    for index, row in tqdm(df_reident.iterrows(), total=df_reident.shape[0]):
        res.append(row['ID'] == row['ID_Compare'])
    df_reident['ID_Match'] = res
    logging.info(f'Identified {len(df_reident[df_reident["ID_Match"] == True])} actual (validated) matches!')
    return df_reident[['Name', 'Gender', 'Age', 'Zip', 'Diagnosis', 'Treatment', 'Outcome', 'ID_Match', 'ID', 'ID_Compare']]

def try_reidentification_noise(df_demographic, df_medical, logger):
    """ Try patient reidentification, compatible with smartnoise-encoded data
    Args:
        df_demographic = demographic data set generated above 
        df_medical = medical data set generated/imported
        k = level of anonymization
    Returns:
        df_dem = demographic data set
    """
    df_reident = pd.DataFrame(columns=df_medical.columns)
    df_demographic_data = df_demographic.copy()
    logging.info(f'Trying reidentification with differential privacy-protected data: {len(df_medical)}. Attacker collection: {len(df_demographic)}')
    logging.info(f'In this setup, we can only count potential matches. We cannot validate for actual matches, as we do not have the unique patient ids after synthesizing.')
    df_demographic_data['Gender_encoded'] = df_demographic_data['Gender_encoded'].astype(str).replace({"0": "F", "1": "M"})
    # Create data
    for index, row in tqdm(df_demographic_data.iterrows(), total=df_demographic.shape[0]):
        df_filtered = df_medical.loc[(df_medical['Gender'] == row['Gender_encoded']) & (df_medical['Age'] == row['Age_encoded']) & (df_medical.Zip == row['Zip_encoded'])].copy()
        if len(df_filtered) == 0:
            continue
        df_filtered = df_filtered.sample(1)
        df_filtered['Age'] = row['Age_encoded']
        df_filtered['Zip'] = row['Zip_encoded']
        df_reident = df_reident.append(df_filtered)
    # Return information about potential matches
    logging.info(f'Identified {len(df_reident)} potential matches!')
    return df_reident

def reident_plot(df_reident, df_medical, col):
    """ Plot distribution of potential, actual and non-matches in a pie chart
    Args:
        df_reident = results of reidentification-attack as df
        df_medical = medical data set generated/imported
        col = column name of ID
    """
    r = pd.DataFrame(df_reident[col].append(pd.Series(['Unknown'] * (len(df_medical) - len(df_reident)))))
    pie, ax = plt.subplots(figsize=[15,9])
    plt.pie(x = r[0].sort_index().value_counts(), autopct="%.1f%%", labels = r[0].map({False: 'Potential Match', True: 'Validated/Actual Match', 'Unknown': 'No Match'}).sort_index().unique(), pctdistance = 0.5, textprops={'fontsize': 14})
    plt.title("Revealed Identities", fontsize = 20)
