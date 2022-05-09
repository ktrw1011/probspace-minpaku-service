# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['racoon',
 'racoon.dataset',
 'racoon.encoder',
 'racoon.encoder._custom',
 'racoon.estimator',
 'racoon.experiment',
 'racoon.runner']

package_data = \
{'': ['*']}

install_requires = \
['catboost>=1.0.4,<2.0.0',
 'category-encoders>=2.3.0,<3.0.0',
 'datasets>=1.18.4,<2.0.0',
 'ipynb-path>=0.1.4,<0.2.0',
 'lightgbm>=3.3.2,<4.0.0',
 'pandas>=1.4.1,<2.0.0',
 'scikit-learn>=1.0.2,<2.0.0',
 'xfeat @ git+https://github.com/pfnet-research/xfeat.git@master',
 'xgboost>=1.5.2,<2.0.0']

entry_points = \
{'console_scripts': ['racoon-exp = racoon.exp:main']}

setup_kwargs = {
    'name': 'racoon',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'ktrw',
    'author_email': 'ktr.w1011@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
