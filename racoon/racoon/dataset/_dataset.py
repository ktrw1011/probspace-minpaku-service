from typing import Optional, Iterable, Tuple, Union, Dict

import numpy as np
from sklearn.utils import multiclass
from datasets import DatasetDict, Dataset

class TableDataset(DatasetDict):
    def __init__(
        self,
        train_features:np.ndarray,
        train_targets:np.ndarray,
        cv:Optional[Iterable[Tuple[np.ndarray, np.ndarray]]]=None,
        test_features:Optional[np.ndarray]=None,
        type_of_target: str = 'auto',
        ):
    
        train_ds = Dataset.from_dict({'data': train_features, 'label': train_targets})
        train_ds.set_format(type='numpy')
        self['train'] = train_ds

        class_size = 1
        if type_of_target == 'auto':
            type_of_target = multiclass.type_of_target(train_targets)
        if type_of_target == 'multiclass':
            class_size = np.unique(train_targets, return_counts=True)[0].size

        self.class_size = class_size
        self.type_of_target = type_of_target

        self.cv = cv

        if test_features is not None:
            test_ds = Dataset.from_dict({'data': test_features})
            test_ds.set_format(type='numpy')
            self['test'] = test_ds

    @property
    def train(self) -> Dataset:
        return self['train']

    @property
    def test(self) -> Optional[Dataset]:
        if 'test' in self:
            return self['test']

    def __repr__(self) -> str:
        repr = super().__repr__()
        repr += f"\nclass_size: {self.class_size} ({self.type_of_target})"
        return repr

    def get_fold(
        self,
        fold:int,
        return_dict:bool=False) -> Union[Tuple[Dict, Dict], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        if self.cv is None:
            raise ValueError

        trn_idx = self.cv[fold][0]
        val_idx = self.cv[fold][1]

        if return_dict:
            trn_item = self['train'][trn_idx]
            val_item = self['train'][val_idx]
            return trn_item, val_item
        else:
            trn_data, trn_label = self['train']['data'][trn_idx], self['train']['label'][trn_idx]
            val_data, val_label = self['train']['data'][val_idx], self['train']['label'][val_idx]
            return (trn_data, trn_label), (val_data, val_label)


    def iter_fold(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        for i in range(len(self.cv)):
            yield self.get_fold(i)