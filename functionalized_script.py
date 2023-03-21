import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel('datasets/online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df_.copy()

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


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains('C', na=False)]
    dataframe = dataframe[dataframe['Quantity'] > 0]
    dataframe = dataframe[dataframe['Price'] > 0]
    replace_with_thresholds(dataframe, 'Quantity')
    replace_with_thresholds(dataframe, 'Price')
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_product(dataframe, StockCode):
    return df.loc[df['StockCode'] == StockCode][['Description']].values[0].tolist()


def create_rules(dataframe, id=True, country = 'France'):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.01)
    return rules

df = retail_data_prep(df)
rules = create_rules(df)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values('lift', ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules['antecedents']):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]['consequents'])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, 22492, 4)
