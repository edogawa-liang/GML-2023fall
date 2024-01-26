import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling


class SubgraphData(object):
    def __init__(self, data, explainer, k=3, train=True):
        '''
        Parameter:
        data: the original graph data
        explainer: GNNExplainer setup
        k: number of nodes to be explained (default: 3)
        train: use for training or val/testing (default: True)
        ================================  
        
        Return:
        `x`: Node feature matrix
        `edge_index`: Adjacency list (training:子圖包含的節點在原圖經過的所有邊, val/testing: 子圖的邊)
        `edge_label`: 欲預測對象之標籤
        `edge_label_index`: 欲預測對象(在原圖內，但不在子圖集合的edge)
        `train`: if True, use for training; else, use for val/testing

        '''
        self.data = data
        self.explainer = explainer
        self.k = k
        self.train = train
    
    def get_data(self):
        self._generate_k_subgraph()
        return self.pygdata

    # GNNExplainer 生成k個子圖
    def _generate_k_subgraph(self):
        print("GNNExplainer 生成子圖中...")
        # randomly select k nodes for explanation
        self.node_id_explain = np.random.choice(self.data.num_nodes, self.k) # the node index we want to explain
        
        # make explanations on the node selected for explanation
        self.explanations = {}
        for n in self.node_id_explain:
            explanation = self.explainer(self.data.x, self.data.edge_index, index=n)
            self.explanations[n] = explanation

        print("子圖生成完畢")
        self._find_nodes_edges_in_subgraph()


    # 找出所有子圖的node和edge
    def _find_nodes_edges_in_subgraph(self):
        all_nodes = torch.empty((0,), dtype=torch.int)
        all_edges = torch.empty((2, 0), dtype=torch.int)

        for id_explain in self.node_id_explain:
            explain = self.explanations[id_explain]
            
            # edge_mask 不為0的位置
            non_zero_positions = (explain.edge_mask != 0).nonzero().squeeze()

            # 找出不為0的位置的edge_index
            edge_index = self.data.edge_index[:, non_zero_positions].reshape(2, -1)
            all_edges = torch.cat((all_edges, edge_index), dim=1)

            # 找出這些edge_index會經過的node
            node_index = torch.cat([edge_index[0], edge_index[1]]).reshape(1, -1).unique()
            all_nodes = torch.cat((all_nodes, node_index), dim=0)

        self.unique_all_nodes = torch.unique(all_nodes)
        self.unique_all_edges = torch.unique(all_edges.t(), dim=0).t()

        self._find_edge_in_origraph()


    # 挑選出在原圖中，與 子圖節點集合中相連的所有邊
    def _find_edge_in_origraph(self):
        edges = self.data.edge_index.t().tolist()
        selected_edges = [edge for edge in edges if edge[0] in self.unique_all_nodes and edge[1] in self.unique_all_nodes]
        self.tensor_selected_edges = torch.tensor(selected_edges).T.unique(dim=1)

        self._check_path()
        

    # 檢查子圖的節點，在原圖中是否存在路徑
    def _check_path(self):
        G = nx.Graph()
        G.add_nodes_from(self.unique_all_nodes.tolist())
        G.add_edges_from(self.tensor_selected_edges.T.tolist())

        if nx.is_connected(G):
            print(f"{self.k}個子圖的所有節點，在原圖中存在路徑!")
            self._sort_nodes()
        else:
            print("重挑子圖！")
            self._generate_k_subgraph()


    # 針對會使用到的node 做重新排序
    def _sort_nodes(self):
        # 創建一個字典來儲存舊標籤到新標籤的映射
        node_mapping = {node.item(): i for i, node in enumerate(torch.unique(self.unique_all_nodes))}

        # 使用映射來更新節點和邊的標籤
        self.unique_all_nodes_sort = torch.tensor([node_mapping[node.item()] for node in self.unique_all_nodes]) # 子圖的所有node
        self.unique_all_edges_sort = torch.tensor([[node_mapping[node.item()] for node in edge] for edge in self.unique_all_edges.t()]).t() # 子圖的所有edge
        self.tensor_selected_edges_sort = torch.tensor([[node_mapping[node.item()] for node in edge] for edge in self.tensor_selected_edges.t()]).t() # 在原圖內，與子圖node有連結的所有edge

        # 新的 feature matrix
        self.new_feature_matrix = self.data.x[self.unique_all_nodes]

        # 在原圖內，但不在子圖集合的 edge
        set1 = set(map(tuple, self.unique_all_edges_sort.t().numpy()))
        set2 = set(map(tuple, self.tensor_selected_edges_sort.t().numpy()))
        self.not_in_subgraph = torch.tensor(list(set2 - set1)).t()

        self._pyg_data()
    
    
    def _generate_edges_labels_wt_neg(self, train=True):
        
        # training 跟testing 在每一張圖都固定負樣本
        # self.negative_samples = negative_sampling(
        #             edge_index= self.tensor_selected_edges_sort, num_nodes=len(self.unique_all_nodes_sort),
        #             num_neg_samples=self.not_in_subgraph.shape[1], method='sparse') 
        
        # negative_labels = torch.zeros((1, self.not_in_subgraph.shape[1]), dtype=torch.long)
        # edge_label = torch.cat((torch.ones((1, self.not_in_subgraph.size(1)), dtype=torch.long), negative_labels), dim=1)[0]
        # edge_label_index = torch.cat((self.not_in_subgraph, self.negative_samples), dim=1)
        # if train:
        #     edge_index = self.tensor_selected_edges_sort
        # else:
        #     edge_index = self.unique_all_edges_sort
    
        if train:
            # training 可以看到圖中所有邊。訓練時再每個batch生成負樣本
            edge_index = self.tensor_selected_edges_sort
            edge_label = torch.ones((1, self.not_in_subgraph.size(1)), dtype=torch.long)[0]
            edge_label_index = self.not_in_subgraph

        else:
            # valid, test 只能看到子圖中的邊。加負樣本
            edge_index = self.unique_all_edges_sort
            
            self.negative_samples = negative_sampling(
            edge_index= self.tensor_selected_edges_sort, num_nodes=len(self.unique_all_nodes_sort),
            num_neg_samples=self.not_in_subgraph.shape[1], method='sparse') 

            negative_labels = torch.zeros((1, self.not_in_subgraph.shape[1]), dtype=torch.long)
            edge_label = torch.cat((torch.ones((1, self.not_in_subgraph.size(1)), dtype=torch.long), negative_labels), dim=1)[0]
            edge_label_index = torch.cat((self.not_in_subgraph, self.negative_samples), dim=1)


        return edge_index, edge_label, edge_label_index
    
    '''
    summary
    子圖的所有node: `unique_all_nodes_sort`
    子圖的所有edge: `unique_all_edges_sort`
    原圖中跟子圖node有連結的所有edge: `tensor_selected_edges_sort`
    新圖的node feature matrix: `new_feature_matrix`
    在原圖內，但不在子圖集合的edge: `not_in_subgraph`
    '''
    
    def _pyg_data(self):
        edge_index, edge_label, edge_label_index= self._generate_edges_labels_wt_neg(self.train)
        self.pygdata = Data(x=self.new_feature_matrix, edge_index=edge_index,
                            edge_label=edge_label, edge_label_index=edge_label_index, 
                            edge_subgraph_index = self.unique_all_edges_sort, train=self.train) 
        
        return self.pygdata