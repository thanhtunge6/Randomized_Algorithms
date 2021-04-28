import bnlearn as bn

# Load asia DAG
model = bn.import_DAG('asia')

# plot ground truth
# G = bn.plot(model)

# Sampling
df = bn.sampling(model, n=10000)
print(df.info())
# Structure learning of sampled dataset
# model_learned = bn.structure_learning.fit(df, methodtype='cl', scoretype='bic', root_node='either')
#
# G = bn.plot(model_learned)