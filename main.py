import bnlearn as bn
from pgmpy.estimators import PC
from data_config import DATA_CONFIG as dc


def get_sample_size(node, epsilon=0.001, t_structure=True):
    if t_structure:
        return int(node/epsilon)
    else:
        return int(node**2/epsilon**2)


dataset = "asia"
model = bn.import_DAG(dc[dataset]["path"])

sample_size = get_sample_size(dc[dataset]["nodes"], epsilon=0.01, t_structure=False)
data = bn.sampling(model, n=sample_size)

# # Structure learning of sampled dataset
# cl_model = bn.structure_learning.fit(data, methodtype='cl', scoretype='bic', root_node='BirthAsphyxia')
#
# # Parameter learning of sampled dataset
# cl_model = bn.parameter_learning.fit(cl_model, data)
# G = bn.plot(cl_model)

pc_model = PC(data).estimate()
print(pc_model.edges())