from sklearn.model_selection import KFold

class LinearRegression(object):
    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500, bs=50, cv=None, 
                 init='zeros', use_momentum=False, momentum=0.9):
        self.lr = lr
        self.num_epochs = num_epochs
        self.bs = bs
        self.method = method
        self.cv = cv if cv is not None else KFold(n_splits=5)
        self.regularization = regularization
        self.init = init
        self.use_momentum = use_momentum
        self.momentum = momentum

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def r2(self, ytrue, ypred):
        ss_res = np.sum((ytrue - ypred) ** 2)
        ss_tot = np.sum((ytrue - np.mean(ytrue)) ** 2)
        return 1 - ss_res / ss_tot

    def _initialize_weights(self, n_features):
        if self.init == 'zeros':
            self.theta = np.zeros(n_features)
        elif self.init == 'xavier':
            limit = np.sqrt(1 / n_features)
            self.theta = np.random.uniform(-limit, limit, size=n_features)
        else:
            raise ValueError("init must be 'zeros' or 'xavier'")

    def fit(self, X_train, y_train):
        self.kfold_scores = []
        self.val_loss_old = np.inf

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]

            self._initialize_weights(X_cross_train.shape[1])

            for epoch in range(self.num_epochs):
                perm = np.random.permutation(X_cross_train.shape[0])
                X_cross_train = X_cross_train[perm]
                y_cross_train = y_cross_train[perm]

                if self.method == 'mini':
                    for batch_idx in range(0, X_cross_train.shape[0], self.bs):
                        X_method_train = X_cross_train[batch_idx:batch_idx+self.bs, :]
                        y_method_train = y_cross_train[batch_idx:batch_idx+self.bs]
                        train_loss = self._train(X_method_train, y_method_train)
                else:
                    X_method_train = X_cross_train
                    y_method_train = y_cross_train
                    train_loss = self._train(X_method_train, y_method_train)

                yhat_val = self.predict(X_cross_val)
                val_loss_new = self.mse(y_cross_val, yhat_val)

                
                # Calculate R2 score
                ss_res = np.sum((y_cross_val - yhat_val) ** 2)
                ss_tot = np.sum((y_cross_val - np.mean(y_cross_val)) ** 2)
                r2_score = 1 - (ss_res / ss_tot)


                if np.allclose(val_loss_new, self.val_loss_old):
                    break
                self.val_loss_old = val_loss_new

            self.kfold_scores.append({'mse': val_loss_new, 'r2': r2_score})
            print(f"Fold {fold}: MSE = {val_loss_new:.4f}, r2 = {r2_score:.4f}")


    def _train(self, X, y):
        m = X.shape[0]
        yhat = self.predict(X)
        grad = (1/m) * X.T @ (yhat - y) + self.regularization.derivation(self.theta)

        if self.use_momentum:
            if not hasattr(self, 'velocity'):
                self.velocity = np.zeros_like(self.theta)
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * grad
            self.theta = self.theta - self.lr * self.velocity
        else:
            self.theta = self.theta - self.lr * grad

        return self.mse(y, yhat)

    def predict(self, X):
        return X @ self.theta

    def _coef(self):
        return self.theta[1:] 

    def _bias(self):
        return self.theta[0] 
        

    def plot_feature_importance(self, feature_names=None):
        coefs = self.theta[1:]  # exclude bias
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(coefs))]
        else:
            feature_names = feature_names[:len(coefs)]  # truncate if extra

        indices = np.argsort(np.abs(coefs))[::-1]
        sorted_coefs = coefs[indices]
        sorted_features = [feature_names[i] for i in indices]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.barh(sorted_features, sorted_coefs)
        plt.xlabel('Coefficient value')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()

#Lasso
class Lasso:
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
    
    def derivation(self, theta):
        return self.l * np.sign(theta)
#Ridge
class Ridge:
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
    
    def derivation(self, theta):
        return self.l * 2 * theta
#Elastic
class Elastic:
    def __init__(self, l, l_ratio):
        self.l = l
        self.l_ratio = l_ratio
        
    def __call__(self, theta): #__call__ allows us to call class as method
        l1 = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2 = (1 - self.l_ratio) * self.l * np.sum(np.square(theta)) 
        return (l1 + l2)
    
    def derivation(self, theta):
        l1 = self.l * self.l_ratio * np.sign(theta)
        l2 = 2 * self.l * (1 - self.l_ratio) * theta
        return (l1 + l2)

#inherits LinearRegression and has separate classes for each of this regularization algorithm

class LassoRegression(LinearRegression):
    def __init__(self, method, lr, l):
        self.regularization = Lasso(l)
        super().__init__(self.regularization, lr, method)

class RidgeRegression(LinearRegression):
    def __init__(self, method, lr, l):
        self.regularization = Ridge(l)
        super().__init__(self.regularization, lr, method)
        
class ElasticRegression(LinearRegression):
    def __init__(self, method, lr, l, l_ratio=0.5):
        self.regularization = Elastic(l, l_ratio)
        super().__init__(self.regularization, lr, method)

class NoRegularization:
    def __call__(self, theta):
        return 0
    def derivation(self, theta):
        return 0