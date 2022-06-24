from torch_geometric.data import Data
from torch_sparse import SparseTensor
import re

class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings.item())
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0
