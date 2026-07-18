import torch


class WhiteningTransform:
    def __init__(self, data=None, whitening_matrix=None, mean=None, n_components=None):
        # 如果 data 是 numpy array，则转换为 torch tensor。
        if data is not None and not isinstance(data, torch.Tensor):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            data = torch.tensor(data, dtype=torch.float32, device=device)

        self.n_components = n_components
        self.fitted = False
        self.truncated_ = False
        self.used_components_ = None
        self.fit(data)
        self.fitted = True

    def to(self, device):
        """将实例移动到指定 device。"""
        self.mean_ = self.mean_.to(device)
        self.whitening_matrix_ = self.whitening_matrix_.to(device)
        if hasattr(self, 'eigenvalues_'):
            self.eigenvalues_ = self.eigenvalues_.to(device)
        if hasattr(self, 'eigenvectors_'):
            self.eigenvectors_ = self.eigenvectors_.to(device)
        return self

    def fit(self, X):
        if self.fitted:
            raise ValueError("实例已经拟合。")
        if not isinstance(X, torch.Tensor):
            raise TypeError("`X` 必须是 torch.Tensor")

        N, D = X.shape
        self.mean_ = X.mean(dim=0)
        X.sub_(self.mean_) # 原地操作以节省内存（X 可能很大），后面会把均值加回去

        cov = torch.cov(X.T)
        eigenvalues_complex, eigenvectors_complex = torch.linalg.eigh(cov)

        eigenvalues = eigenvalues_complex.real
        eigenvectors = eigenvectors_complex.real
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        max_rank = min(N - 1, D)
        self.truncated_ = max_rank < D
        eigenvalues = eigenvalues[:max_rank]
        eigenvectors = eigenvectors[:, :max_rank]

        if self.n_components is not None:
            r = min(self.n_components, max_rank)
            eigenvalues = eigenvalues[:r]
            eigenvectors = eigenvectors[:, :r]
        else:
            r = max_rank

        self.used_components_ = r

        # 白化前过滤无效（非正）特征值
        # --------------------------------------------------------------
        # 理论上协方差矩阵是半正定的，因此所有特征值 λ ≥ 0。
        # 实践中可能出现很小的负特征值，常见原因包括：
        #   • 浮点舍入误差（尤其是 float32 / float16），
        #   • C = XᵀX / (N−1) 未完全对称化时产生的微小非对称，
        #   • mixed precision 或 AMP 中累积的精度损失，
        #   • 病态数据中的极小数值噪声。
        #
        # 若对 λ ≤ 0 计算 1/√λ，会在白化矩阵中产生 NaN 或 Inf。
        # 因此在求逆前屏蔽所有零或负特征值。这通常不会发生，但能防御少见边界情况。

        valid_mask = eigenvalues > 0
        if not torch.any(valid_mask):
            raise ValueError("All eigenvalues are too small or non-positive.")

        # 只保留稳定的特征向量/特征值。
        eigenvalues = eigenvalues[valid_mask]
        eigenvectors = eigenvectors[:, valid_mask]


        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        diag_mat = torch.diag(1.0 / torch.sqrt(eigenvalues + 1e-5))

        self.whitening_matrix_ = eigenvectors @ diag_mat

        # 把均值加回 X。
        X.add_(self.mean_)  # 原地操作以节省内存

    def transform_numpy(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32,device=self.mean_.device)
        return self.transform(X).numpy()

    def transform(self, X):
        if not self.fitted:
            raise ValueError("请先调用 fit")
        device = self.mean_.device
        X = X.to(device)
        X_centered = X - self.mean_
        return X_centered @ self.whitening_matrix_

    def get_eigenvalues(self):
        """
        获取白化变换中使用的特征值。
        
        返回：
            torch.Tensor: 降序排列的特征值，形状为 (d,)，其中 d 是使用的成分数。
        
        抛出：
            ValueError: 如果该 transform 尚未拟合。
        """
        if not self.fitted:
            raise ValueError("访问特征值前必须先拟合 transform")
        return self.eigenvalues_

    def get_eigenvectors(self):
        """
        获取白化变换中使用的特征向量。
        
        返回：
            torch.Tensor: 降序排列的特征向量，形状为 (D, d)，其中 d 是使用的成分数。
        
        抛出：
            ValueError: 如果该 transform 尚未拟合。
        """
        if not self.fitted:
            raise ValueError("访问特征向量前必须先拟合 transform")
        return self.eigenvectors_

    def truncation_info(self):
        """
        返回：
            truncated (bool): 若发生 rank 截断则为 True。
            used_components (int): 最终保留的 eigen-vector 数量。
        """
        return self.truncated_, self.used_components_
