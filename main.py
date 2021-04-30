import bnlearn as bn
from pgmpy.estimators import PC, TreeSearch
from pgmpy.base import DAG
import networkx as nx
import matplotlib.pyplot as plt
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

est = TreeSearch(data, root_node='either')
cl_model = est.estimate(estimator_type='chow-liu')

pc_model = PC(data).estimate()

nx.draw_circular(pc_model, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)
plt.show()