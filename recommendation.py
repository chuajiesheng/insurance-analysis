import csv
import copy

INPUT_FILE = './input/Data.csv'
rows = []
# read to a array (data set not big, no performance issue)
with open(INPUT_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

insured = 'INSURED_ID'
policy_owners = set(map(lambda row: row[insured], rows))
print('# Policy Owner: {}'.format(len(policy_owners)))

EMPTY_POLICY_GROUP = {
    'Health': 0,
    'Protection': 0,
    'Investment': 0,
    'Retirement': 0,
    'Savings': 0
}

current_policies = dict()
for r in rows:
    person = r[insured]
    if person not in current_policies:
        current_policies[person] = copy.deepcopy(EMPTY_POLICY_GROUP)

    policy_category = r['PRODUCT_CATEGORY']
    current_policies[person][policy_category] += 1

# find people with everything
people_with_all_types = 0
for key in current_policies:
    if current_policies[key]['Health'] > 0 and \
                    current_policies[key]['Protection'] > 0 and \
                    current_policies[key]['Investment'] > 0 and \
                    current_policies[key]['Retirement'] > 0 and \
                    current_policies[key]['Savings'] > 0:
        print('{} have all the policy types'.format(key))
        people_with_all_types += 1

# i am expecting one person that won't have a good recommendation
print('# People with all policy categories: {}'.format(people_with_all_types))

CATEGORIES = ['Health', 'Protection', 'Investment', 'Retirement', 'Savings']
INTERESTED_FIELDS = ['MEDICAL_FLAG', 'GENDER', 'AGE_GROUP', 'Health', 'Protection', 'Investment', 'Retirement', 'Savings']
PERSONA = dict()
for f in INTERESTED_FIELDS:
    PERSONA[f] = 0


def get_dict(dataset):
    d = copy.deepcopy(PERSONA)

    d['MEDICAL_FLAG'] = dataset['MEDICAL_FLAG'] == 'Y'
    d['GENDER'] = dataset['GENDER']
    d['AGE_GROUP'] = (int(dataset['ENTRY_AGE']) % 10) * 10

    return d

# read and tabulate everyone policies groupings
people = dict()
for r in rows:
    person = r[insured]
    if person not in people:
        people[person] = get_dict(r)

    policy_category = r['PRODUCT_CATEGORY']
    people[person][policy_category] += 1

# transform into matrix for clustering
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
X = v.fit_transform(people.values())

# cluster into 5 cluster, because of 5 different product cataegories
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5).fit(X)

PREDICT_FILE = './input/upsell_crosssell.csv'
for key in people.keys():
    data = people[key]
    x = v.transform(data)
    k = kmeans.predict(x)
    dict_array = v.inverse_transform(kmeans.cluster_centers_[k][0])
    center = dict_array[0]

    diff = dict()
    for c in CATEGORIES:
        diff[c] = center[c] - data[c]

    result = min(diff, key=diff.get)

    with open(PREDICT_FILE, 'a') as f:
        f.write('{},{}\n'.format(key, result))
