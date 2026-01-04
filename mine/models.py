import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=128):
        super(SimpleMINE, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, y):
        # Nối x và y để đưa vào mạng
        combined = torch.cat([x, y], dim=1)
        return self.layers(combined)

    def compute_mi(self, x, y):
        """
        Hàm tính toán giá trị MI (Mutual Information)
        """
        # 1. Cặp thật (Joint): x_i đi cùng y_i
        t_joint = self.forward(x, y)
        
        # 2. Cặp giả (Marginal): x_i đi cùng y_j (y bị xáo trộn)
        y_shuffled = y[torch.randperm(y.size(0))]
        t_margin = self.forward(x, y_shuffled)
        
        # 3. Công thức MINE (Donsker-Varadhan bound)
        # MI = mean(t_joint) - log(mean(exp(t_margin)))
        mi_estimate = torch.mean(t_joint) - (torch.logsumexp(t_margin, dim=0) - torch.log(torch.tensor(y.size(0), dtype=torch.float)))
        
        return mi_estimate
