############################################
# ASSOCIATION RULE LEARNING
############################################

# 1. Data Preprocessing
# 2. Preparation of ARL Data Structure (Invoice-Product Matrix)
# 3. Extraction of Association Rules
# 4. Providing Product Recommendations to Users at the Cart Stage

############################################
# 1. Data Preprocessing
############################################

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel('datasets/online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df_.copy()

df.head()
df.shape
df.isnull().sum()
df.dropna(inplace=True)

df = df[~df['Invoice'].str.contains('C', na=False)]

df.describe().T

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

############################################
# 2. Preparation of ARL Data Structure (Invoice-Product Matrix)
############################################

df_fr = df.loc[df["Country"] == "France"]

# df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"})
# df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
# df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]
# df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]
# df_fr.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# alternative way:
# basket = df.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0)
# def encode_units(x):
#     if x <= 0:
#         return 0
#     if x >= 1:
#         return 1
# basket_sets = basket.applymap(encode_units)


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'StockCode'])['Description'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)

def check_product(dataframe, StockCode):
    return df.loc[df["StockCode"] == StockCode][["Description"]].values[0].tolist()

check_product(df_fr, 21086) # ['SET/6 RED SPOTTY PAPER CUPS']
check_product(df_fr, 10120) # ['DOGGY RUBBER']

############################################
# 3. Extraction of Association Rules
############################################

frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)

frequent_itemsets.sort_values('support', ascending=False)

rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.01)

rules.head()

#   antecedents consequents  antecedent support  consequent support   support  confidence       lift  leverage  conviction
# 0     (10002)     (21791)            0.020566            0.028278  0.010283    0.500000  17.681818  0.009701    1.943445
# 1     (21791)     (10002)            0.028278            0.020566  0.010283    0.363636  17.681818  0.009701    1.539111
# 2     (10002)     (21915)            0.020566            0.069409  0.010283    0.500000   7.203704  0.008855    1.861183
# 3     (21915)     (10002)            0.069409            0.020566  0.010283    0.148148   7.203704  0.008855    1.149771
# 4     (10002)     (22551)            0.020566            0.136247  0.010283    0.500000   3.669811  0.007481    1.727506


# Antecedents: In the context of association rule learning, antecedents refer to the items or itemsets that occur together frequently in a dataset.
# They are the "if" part of the rule, and represent the conditions or patterns that are observed in the data.
# For example, if we are analyzing a dataset of customer transactions at a grocery store,
# an antecedent might be a set of items that are frequently purchased together, such as "bread and butter".

# Consequents: In association rule learning, consequents refer to the items or itemsets that tend to occur with the antecedents.
# They are the "then" part of the rule, and represent the predicted outcome or result of the pattern observed in the data.
# For example, continuing with the grocery store example, a consequent might be a product that
# is often purchased together with the antecedent, such as "jam".

# Antecedent Support: The ratio of the total number of occurrences of the antecedents in all transactions to the total number of transactions.

# Consequent Support: The ratio of the total number of occurrences of the consequents in all transactions to the total number of transactions.

# Support: The ratio of the number of transactions where both antecedents and consequents occur to the total number of transactions.

# Confidence: The probability that consequents will occur given the occurrence of antecedents in a transaction.
# It is calculated as "confidence(A->B) = support(A&B) / support(A)".

# Lift: The measure of the high-frequency association between antecedents and consequents.
# It is calculated as "lift(A->B) = support(A&B) / (support(A) * support(B))".

# Leverage: The measure of how much more often antecedents and consequents occur together than would be expected by chance.
# It is calculated as "leverage(A->B) = support(A&B) - support(A) * support(B)".

# Conviction: The ratio of the number of times the consequents do not occur given the occurrence of antecedents
# to the number of times it would be expected not to occur.
# It is calculated as "conviction(A->B) = (1 - support(B)) / (1 - confidence(A->B))".

rules[(rules['support'] > 0.05) & (rules['confidence'] > 0.1)] # [323 rows x 9 columns]
rules[(rules['support'] > 0.05) & (rules['confidence'] > 0.1) & (rules['lift'] > 5)] # [84 rows x 9 columns]

check_product(df_fr, 21080) # ['SET/20 RED RETROSPOT PAPER NAPKINS ']
check_product(df_fr, 21086) # ['SET/6 RED SPOTTY PAPER CUPS']


rules[(rules['support'] > 0.05) & (rules['confidence'] > 0.1) & (rules['lift'] > 5)].sort_values('confidence', ascending=False).head()

#                  antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction
# 23707         (21080, 21094)     (21086)            0.102828            0.138817  0.100257    0.975000  7.023611  0.085983   34.447301
# 23706         (21080, 21086)     (21094)            0.102828            0.128535  0.100257    0.975000  7.585500  0.087040   34.858612
# 108820  (21080, POST, 21086)     (21094)            0.084833            0.128535  0.082262    0.969697  7.544242  0.071358   28.758355
# 108822  (21080, POST, 21094)     (21086)            0.084833            0.138817  0.082262    0.969697  6.985410  0.070486   28.419023
# 1777                 (21094)     (21086)            0.128535            0.138817  0.123393    0.960000  6.915556  0.105550   21.529563

check_product(df_fr, 21080) # ['SET/20 RED RETROSPOT PAPER NAPKINS ']
check_product(df_fr, 21094) # ['SET/6 RED SPOTTY PAPER PLATES']

# If a customer has purchased the 2 products above, there is a 97 percent probability that he will buy the product below. (confidence=0.975000)

check_product(df_fr, 21086) # ['SET/6 RED SPOTTY PAPER CUPS']

############################################
# 5. Providing Product Recommendations to Users at the Cart Stage
############################################

sorted_rules = rules.sort_values('lift', ascending=False)

# example product id = 22492
check_product(df_fr, 22492) # ['MINI PAINT SET VINTAGE ']

product_id = 22492
recommendation_list = []
for i, product in enumerate(sorted_rules['antecedents']):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]['consequents'])[0])

recommendation_list[0:3] # [22556, 22551, 22326]

check_product(df_fr, 22556) # ['ROUND SNACK BOXES SET OF4 WOODLAND ']
check_product(df_fr, 22551) # ['PLASTERS IN TIN CIRCUS PARADE ']
check_product(df_fr, 22326) # ['SET OF 4 KNICK KNACK TINS LONDON ']



product_id = 22492
recommendation_list = []
for i, product in enumerate(sorted_rules['antecedents']):
    for j in list(product):
        if j == product_id:
            if len(sorted_rules.iloc[i]['consequents']) == 1:
                recommendation_list.append(list(sorted_rules.iloc[i]['consequents']))

new_recommendation_list = list(set(map(tuple,recommendation_list)))
print(new_recommendation_list) # [(22557,), (22139,), (22029,), (23206,), (22333,)...]

new_recommendation_list = [i[0] for i in new_recommendation_list]
print(new_recommendation_list) # [22557, 22139, 22029, 23206, 22333...]

check_product(df_fr, 22492) # ['MINI PAINT SET VINTAGE ']

check_product(df_fr, 22557) # ['PLASTERS IN TIN VINTAGE PAISLEY ']
check_product(df_fr, 22139) # ['RETROSPOT TEA SET CERAMIC 11 PC ']

