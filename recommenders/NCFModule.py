import torch
import torch.nn as nn


class NCFRecommender(nn.Module):

    def __init__(self, n_users, n_items, factors, layers, p=0.2):
        super().__init__()

        self.user_gmf = nn.Embedding(n_users, factors)
        self.item_gmf = nn.Embedding(n_items, factors)

        self.user_mlp = nn.Embedding(n_users, int(layers[0]/2))
        self.item_mlp = nn.Embedding(n_items, int(layers[0]/2))

        self.layerlist = nn.ModuleList()
        for i in range(0, len(layers) - 1):
            self.layerlist.append(nn.Linear(layers[i], layers[i + 1]))
            self.layerlist.append(nn.ReLU())

        self.mlp_layers = nn.Sequential(*self.layerlist)
        self.predict = nn.Linear(factors + layers[-1], 1)

        self.init_weight()

    def init_weight(self):
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
        user_gmf = self.user_gmf(user)
        item_gmf = self.item_gmf(item)
        output_gmf = torch.mul(user_gmf, item_gmf)

        user_mlp = self.user_mlp(user)
        item_mlp = self.item_mlp(item)
        output_mlp = self.mlp_layers(torch.cat((user_mlp, item_mlp), -1))

        y_pred = self.predict(torch.cat((output_gmf, output_mlp), -1))
        return y_pred.squeeze()
