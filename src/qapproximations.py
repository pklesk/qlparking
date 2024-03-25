import numpy as np
from numba import jit
from numba import int32, float64, boolean
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
import json
import sys

class QMLPRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_actions, hidden_layer_sizes, n_steps=1, batch_size=128, learning_rate=1e-4, random_state=0, ema_decay=0.0):
        self.n_actions = n_actions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_steps = n_steps
        self.ema_decay = ema_decay
        self.batch_size = batch_size 
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_actions={self.n_actions}, "
        s += f"hidden_layer_sizes={self.hidden_layer_sizes}, " 
        s += f"n_steps={self.n_steps}, "
        s += f"batch_size={self.batch_size}, "
        s += f"learning_rate={self.learning_rate}, "
        s += f"random_state={self.random_state}, "
        s += f"ema_decay: {self.ema_decay})"
        return s
        
    def __repr__(self):
        return self.__str__()      
        
    def fit(self, X, y, actions_taken=None):
        if self.ema_decay > 0.0:
            last_coefs = []
            last_intercepts = []
            if hasattr(self, "mlps_"):
                for a in range(self.n_actions):
                    a_coefs = []
                    a_intercepts = []
                    for l in range(len(self.hidden_layer_sizes)):
                        a_coefs.append(np.copy(self.mlps_[a].coefs_[l]))
                        a_intercepts.append(np.copy(self.mlps_[a].intercepts_[l]))
                    last_coefs.append(a_coefs)
                    last_intercepts.append(a_intercepts)                
        if not hasattr(self, "mlps_"):
            self.mlps_ = []
            for a in range(self.n_actions):
                self.mlps_.append(MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.n_steps, batch_size=self.batch_size, learning_rate_init=self.learning_rate, random_state=self.random_state, warm_start=True))
        m = X.shape[0]                    
        if actions_taken is None:
            indexes = np.arange(m)
        for a in range(self.n_actions):
            if actions_taken is not None:
                indexes = np.where(actions_taken == a)[0]  
            if indexes.size > 0:
                X_sub = X[indexes]
                y_sub = y[indexes, a]                
                self.mlps_[a].fit(X_sub, y_sub)
            else:
                self.mlps_[a].coefs_ = last_coefs[a]
                self.mlps_[a].intercepts_ = last_intercepts[a]
        if self.ema_decay > 0.0 and len(last_coefs) > 0:
            for a  in range(self.n_actions):
                for l in range(len(self.hidden_layer_sizes)):                            
                    self.mlps_[a].coefs_[l] = self.ema_decay * last_coefs[a][l] + (1.0 - self.ema_decay) * self.mlps_[a].coefs_[l]
                    self.mlps_[a].intercepts_[l] = self.ema_decay * last_intercepts[a][l] + (1.0 - self.ema_decay) * self.mlps_[a].intercepts_[l]             
                                    
    def predict(self, X):  
        m = X.shape[0]
        y = np.zeros((m, self.n_actions))
        for a in range(self.n_actions):
            y[:, a] = self.mlps_[a].predict(X)
        return y
    
    def average_with_other(self, other, fraction_other):
        for a  in range(self.n_actions):
            for l in range(len(self.hidden_layer_sizes)):                            
                self.mlps_[a].coefs_[l] = (1.0 - fraction_other) * self.mlps_[a].coefs_[l] + fraction_other * other.mlps_[a].coefs_[l]
                self.mlps_[a].intercepts_[l] = (1.0 - fraction_other) * self.mlps_[a].intercepts_[l] + fraction_other * other.mlps_[a].intercepts_[l]
               
    def json_dump(self, fname):
        d = {}
        d["__class__.__name__"] = self.__class__.__name__
        d["n_actions"] = self.n_actions
        d["hidden_layer_sizes"] = self.hidden_layer_sizes
        d["n_steps"] = self.n_steps
        d["ema_decay"] = self.ema_decay
        mlps_to_dump = []
        if hasattr(self, "mlps_"):            
            for a  in range(self.n_actions):
                mlp_a_to_dump = []
                for l in range(len(self.hidden_layer_sizes)):
                    mlp_a_to_dump.append({"coefs_": self.mlps_[a].coefs_[l].tolist(), "intercepts_": self.mlps_[a].intercepts_[l].tolist()})
                mlps_to_dump.append(mlp_a_to_dump)
        d["mlps_"] = mlps_to_dump
        try:
            f = open(fname, "w+")
            json.dump(d, f, indent=2)
            f.close()
        except IOError:
            sys.exit(f"[error occurred when trying to dump regressor as json to file: {fname}]")
            
    @staticmethod
    def json_load(fname):        
        try:
            f = open(fname, "r")
            d = json.load(f)
            params = {}
            params["n_actions"] = d["n_actions"]
            params["hidden_layer_sizes"] = d["hidden_layer_sizes"]
            params["n_steps"] = d["n_steps"]
            params["ema_decay"] = np.float64(d["ema_decay"])
            regressor = QMLPRegressor(**params)
            regressor.mlps_ = []
            mlps_loaded = d["mlps_"]
            for a  in range(regressor.n_actions):
                regressor.mlps_.append(MLPRegressor(hidden_layer_sizes=regressor.hidden_layer_sizes, max_iter=regressor.n_steps, warm_start=True, batch_size=128, learning_rate_init=1e-4, random_state=0))
                regressor.mlps_[a].coefs_ = []
                regressor.mlps_[a].intercepts_ = []
                for l in range(len(regressor.hidden_layer_sizes)):
                    regressor.mlps_[a].coefs_.append(np.array(mlps_loaded[a][l]["coefs_"]))
                    regressor.mlps_[a].intercepts_.append(np.array(mlps_loaded[a][l]["intercepts_"]))
            f.close()
        except IOError:
            sys.exit(f"[error occurred when trying to load regressor from json file: {fname}]")    
        return regressor            

class QRidgeRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_actions, l2_penalty=1e-7, fit_intercept=True, sample_size_factor=True, use_numba=False, ema_decay=0.0):
        self.n_actions = n_actions
        self.l2_penalty = l2_penalty
        self.fit_intercept = fit_intercept
        self.sample_size_factor = sample_size_factor
        self.use_numba = use_numba
        self.ema_decay = ema_decay
        
    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_actions={self.n_actions}, "
        s += f"l2_penalty={self.l2_penalty}, " 
        s += f"fit_intercept={self.fit_intercept}, "
        s += f"sample_size_factor: {self.sample_size_factor}, "
        s += f"use_numba: {self.use_numba}, "
        s += f"ema_decay: {self.ema_decay})"
        return s        
        
    def fit(self, X, y, actions_taken=None):
        m, n = X.shape
        if hasattr(self, "coefs_"):
            last_coefs = self.coefs_
            last_intercepts = self.intercepts_
        else:            
            last_coefs = np.zeros((self.n_actions, n))
            last_intercepts = np.zeros(self.n_actions)                
        self.coefs_ = np.zeros((self.n_actions, n))
        self.intercepts_ = np.zeros(self.n_actions)        
        X_ext = np.copy(np.c_[np.ones((m, 1)), X]) if self.fit_intercept else X        
        if self.use_numba:
            result = QRidgeRegressor.fit_numba(X_ext, y, actions_taken, last_coefs, last_intercepts, self.n_actions, self.l2_penalty, self.fit_intercept, self.sample_size_factor)
            if self.fit_intercept:
                self.intercepts_ = result[:, 0]
                self.coefs_ = result[:, 1:]                  
            else:
                self.intercepts_ = np.zeros(self.n_actions)
                self.coefs_ = result
        else:
            self.fit_numpy(X_ext, y, actions_taken, last_coef, last_intercept)
        if self.ema_decay > 0.0:
            self.coefs_ = self.ema_decay * last_coefs + (1.0 - self.ema_decay) * self.coefs_
            self.intercepts_ = self.ema_decay * last_intercepts + (1.0 - self.ema_decay) * self.intercepts_ 

    @staticmethod
    @jit(float64[:, :](float64[:, :], float64[:, :], int32[:], float64[:, :], float64[:], int32, float64, boolean, boolean), nopython=True)
    def fit_numba(X_ext, y, actions_taken, last_coefs, last_intercepts, self_n_actions, self_l2_penalty, self_fit_intercept, self_sample_size_factor):
        m, n = X_ext.shape
        penalty_matrix = self_l2_penalty * np.identity(n)
        result = np.zeros((self_n_actions, n), dtype=float64)
        if self_fit_intercept:
            penalty_matrix[0, 0] = 0.0
        if actions_taken is None:
            indexes = np.arange(m)
        for a in range(self_n_actions):
            if actions_taken is not None:
                indexes = np.where(actions_taken == a)[0]  
            if indexes.size > 0:
                X_sub = X_ext[indexes]
                m_sub = X_sub.shape[0]
                X_sub_T = X_sub.T
                y_sub = y[indexes, a]
                if self_sample_size_factor:
                    penalty_matrix_sub = m_sub * penalty_matrix
                coefs = (np.linalg.inv(X_sub_T.dot(X_sub) + penalty_matrix_sub)).dot(X_sub_T.dot(y_sub))
                if self_fit_intercept:
                    result[a, 1:] = coefs[1:] 
                    result[a, 0] = coefs[0]
                else:
                    result[a] = coefs
            else:
                if last_coefs is not None:
                    result[a, 1:] = last_coefs[a]
                if last_intercepts is not None:
                    result[a, 0] = last_intercepts[a]    
        return result
            
    def fit_numpy(self, X_ext, y, actions_taken=None, last_coef=None, last_intercept=None):
        m, n = X_ext.shape
        penalty_matrix = self.l2_penalty * np.identity(n)
        if self.fit_intercept:
            penalty_matrix[0, 0] = 0.0
        if actions_taken is None:
            indexes = np.arange(m)
        for a in range(self.n_actions):
            if actions_taken is not None:
                indexes = np.where(actions_taken == a)[0]  
            if indexes.size > 0:
                X_sub = X_ext[indexes]
                m_sub = X_sub.shape[0]
                X_sub_T = X_sub.T
                y_sub = y[indexes, a] if len(y.shape) == 2 else y[indexes]
                if self.sample_size_factor:
                    penalty_matrix_sub = m_sub * penalty_matrix
                coef = (np.linalg.inv(X_sub_T.dot(X_sub) + penalty_matrix_sub)).dot(X_sub_T.dot(y_sub))
                if self.fit_intercept:
                    self.coefs_[a] = coef[1:] 
                    self.intercepts_[a] = coef[0]
                else:
                    self.coefs_[a] = coef
            else:
                if last_coefs is not None:
                    self.coefs_[a] = last_coef[a]
                if last_intercepts is not None:
                    self.intercepts_[a] = last_intercept[a]    
                                    
    def predict(self, X):  
        y_pred = X.dot(self.coefs_.T)
        if self.fit_intercept:
            y_pred += self.intercepts_
        return y_pred
    
    def average_with_other(self, other, fraction_other):
        self.coefs_ = (1.0 - fraction_other) * self.coefs_ + fraction_other * other.coefs_
        self.intercepts_ = (1.0 - fraction_other) * self.intercepts_ + fraction_other * other.intercepts_    
        
    def json_dump(self, fname):
        d = {}
        d["__class__.__name__"] = self.__class__.__name__
        d["n_actions"] = self.n_actions
        d["l2_penalty"] = self.l2_penalty
        d["fit_intercept"] = self.fit_intercept
        d["sample_size_factor"] = self.sample_size_factor
        d["use_numba"] = self.use_numba
        d["ema_decay"] = self.ema_decay                
                
        if hasattr(self, "coefs_"):            
            d["coefs_"] = self.coefs_.tolist()
        if hasattr(self, "intercepts_"):            
            d["intercepts_"] = self.intercepts_.tolist()            
        try:
            f = open(fname, "w+")
            json.dump(d, f, indent=2)
            f.close()
        except IOError:
            sys.exit(f"[error occurred when trying to dump regressor as json to file: {fname}]")
            
    @staticmethod
    def json_load(fname):        
        try:
            f = open(fname, "r")
            d = json.load(f)
            params = {}
            params["n_actions"] = d["n_actions"]
            params["l2_penalty"] = d["l2_penalty"]
            params["fit_intercept"] = d["fit_intercept"]
            params["ema_decay"] = np.float64(d["sample_size_factor"])
            params["use_numba"] = d["use_numba"]
            params["ema_decay"] = np.float64(d["ema_decay"])
                        
            regressor = QRidgeRegressor(**params)
            if "coefs_" in d:
                regressor.coefs_ = np.array(d["coefs_"])
            if "intercepts_" in d:
                regressor.intercepts_ = np.array(d["intercepts_"])                
            f.close()
        except IOError:
            sys.exit(f"[error occurred when trying to load regressor from json file: {fname}]")    
        return regressor