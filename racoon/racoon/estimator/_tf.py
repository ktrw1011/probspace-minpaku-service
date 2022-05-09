# from typing import Tuple, Optional, Dict
# import numpy as np
# import tensorflow as tf
# from sklearn.base import BaseEstimator

# class NeuralNet(BaseEstimator):
#     def __init__(self, model:tf.keras.Model):
#         self.model = model

#     def fit(self,
#         X:np.ndarray,
#         y:np.ndarray,
#         eval_set:Tuple[np.ndarray, np.ndarray],
#         params:Optional[Dict]=None,
#         ) -> None:

#         self.model.fit(
#             X=X,
#             y=y,
#             validation_data=eval_set,
#             **params,
#         )

#     def predict_proba(self, X:np.ndarray) -> np.ndarray:
#         return self.model.predict(X)

#     def predict(self, X:np.ndarray) -> np.ndarray:
#         return self.model.predict(X)