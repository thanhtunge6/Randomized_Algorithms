import bnlearn as bn
from data_config import DATA_CONFIG as dc
from scipy.stats import entropy
from collections import Counter


def get_distribution(data, columns):
    data["combination"] = data.apply(lambda row: "".join(["{}".format(row[col]) for col in columns]), axis=1)
    all_combinations = list(data["combination"])
    return Counter(all_combinations)


def get_kl_divergence(x_distribution, y_distribution):
    x_distribution_list = []
    y_distribution_list = []
    for key in y_distribution.keys():
        x_distribution_list.append(x_distribution.get(key, 0))
        y_distribution_list.append(y_distribution.get(key, 0))
    norm_x = [float(i) / sum(x_distribution_list) for i in x_distribution_list]
    norm_y = [float(i) / sum(y_distribution_list) for i in y_distribution_list]
    return entropy(norm_x, qk=norm_y)


# Load asia DAG
dataset = "child"
model = bn.import_DAG(dc[dataset]["path"])

# plot ground truth
# G = bn.plot(model)
sample_size = [int(1000*100**(t/9)) for t in range(0, 10)]
print(sample_size)
# Sampling
true_df = bn.sampling(model, n=100000)
columns = [col for col in true_df.columns]
print("Get true distribution")
true_distribution = get_distribution(true_df, columns)
kl_divergence = dict()

size = sample_size[6]
train_df = true_df.sample(n=size)

# Structure learning of sampled dataset
model_learned = bn.structure_learning.fit(true_df, methodtype='cl', scoretype='bic', root_node='BirthAsphyxia',
                                          verbose=0)
# model_learned = bn.structure_learning.fit(train_df, methodtype='cl', scoretype='bic', root_node='either', verbose=0)

# Parameter learning of sampled dataset
model_learned = bn.parameter_learning.fit(model_learned, train_df, verbose=0)

learn_df = bn.sampling(model_learned, n=100000)
print("Get learned distribution")
learn_distribution = get_distribution(learn_df, columns)

print("Computing KL-divergence...")
print(get_kl_divergence(learn_distribution, true_distribution))

cl_asia_result = {1000: 0.8373080232272676, 1668: 0.4682100823235455, 2782: 0.25921838487679877,
                  4641: 0.14016847946515232, 7742: 0.07833476649070305, 12915: 0.04122673216363457,
                  21544: 0.021289417255265158, 35938: 0.011792981841819952, 59948: 0.006240634422594821,
                  100000: 0.002945903290298712}
cl_child_result = {1000: 2.923811065264134, 1668: 2.0614189883914804, 2782: 1.658959039565084, 4641: 1.356356165032905,
                   7742: 1.220737020139554, 12915: 1.111645388346174}
