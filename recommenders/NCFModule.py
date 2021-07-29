import torch
import torch.nn as nn


class NCFRecommender(nn.Module):

    def __init__(self, n_users, n_items, factors, layers, p=0.2):
        """
        NCF Neural module
        :param n_users: number of users
        :param n_items: number of items
        :param factors: number of factors used
        :param layers: list with integers that corresponds to the the number of neurons in each layer
        :param p: dropout probability
        """
        super().__init__()

        # setting GMF embeddings
        self.user_gmf = nn.Embedding(n_users, factors)
        self.item_gmf = nn.Embedding(n_items, factors)

        # setting MLP embeddings
        self.user_mlp = nn.Embedding(n_users, int(layers[0]/2))
        self.item_mlp = nn.Embedding(n_items, int(layers[0]/2))

        # create layers for MLP part of the ensemble
        self.layerlist = nn.ModuleList()
        for i in range(0, len(layers) - 1):
            self.layerlist.append(nn.Linear(layers[i], layers[i + 1]))
            self.layerlist.append(nn.ReLU())
            self.layerlist.append(nn.BatchNorm1d(layers[i + 1]))
            self.layerlist.append(nn.Dropout(p))

        self.mlp_layers = nn.Sequential(*self.layerlist)
        self.predict = nn.Linear(factors + layers[-1], 1)

        self.init_weight()

    def init_weight(self):
        """
        Function that set the weights for the embeddings and layers
        """
        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)

        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)

        for m in self.layerlist:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.predict.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        """
        Function that passes a user and item embedding to the module
        :param user: torch tensor of users
        :param item: torch tensor of items
        :return: predictions torch tensor
        """

        # get user and item GMF embeddings and multiply them. That is the GMF part
        user_gmf = self.user_gmf(user)
        item_gmf = self.item_gmf(item)
        output_gmf = torch.mul(user_gmf, item_gmf)

        # get the user and item MLP embeddings and pass them thorough the network. That is the MLP part
        user_mlp = self.user_mlp(user)
        item_mlp = self.item_mlp(item)
        output_mlp = self.mlp_layers(torch.cat((user_mlp, item_mlp), -1))

        # concatenate the GMF and MLP outputs and pass them into the last layer
        y_pred = self.predict(torch.cat((output_gmf, output_mlp), -1))
        return y_pred.squeeze()
