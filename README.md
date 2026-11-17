# machine-learning-notebooks

Jupyter notebooks from my machine learning studies (2023): five small projects and a set of exercises on the core algorithms. The course was Udacity's *Intro to Machine Learning with PyTorch*. The project templates for CIFAR-10, Finding Donors and Student Admissions come from the course; the models, training code and analysis are my own.

The notebooks are kept as they were run, including their outputs.

## Projects

| Project | What it does | Result in the notebook |
|---|---|---|
| [CIFAR-10 image classifier](projects/cifar10-image-classifier) | CNN in PyTorch (4 conv layers with batch norm and dropout, 3 dense layers), Adam, learning-rate scheduler | 92 % on a sample of 50 test images (full test-set accuracy was not saved) |
| [Animal image classification](projects/animal-image-classification) | Transfer learning with a pretrained ResNet-18 on the Animals-10 dataset (10 classes) | sample predictions on test images |
| [Finding donors](projects/finding-donors) | Predicts whether a person earns more than $50K from census data; compares decision tree, random forest and linear SVM, tunes with `GridSearchCV`, checks feature importance | accuracy 0.843, F0.5 0.687 |
| [Fake news classification](projects/fake-news-classification) | Text cleaning with NLTK (tokenizing, lemmatizing), bag-of-words features for title and text, feed-forward network in PyTorch | 99.4 % test accuracy; this dataset is known to be easy to separate, so the number says little about real-world fake-news detection |
| [Student admissions](projects/student-admissions) | Two-layer neural network written from scratch in NumPy (one-hot encoding, scaling, backpropagation) | 0.675 accuracy |

## Fundamentals

- [supervised](fundamentals/supervised): linear, multiple and polynomial regression, logistic regression, perceptron, gradient descent, decision tree, naive Bayes, SVM, ensemble methods (AdaBoost), grid search, classification and regression metrics
- [neural-network-math](fundamentals/neural-network-math): errors, cross entropy, weights and biases, layer calculation, backpropagation
- [unsupervised](fundamentals/unsupervised): DBSCAN, Gaussian mixture models, hierarchical clustering, customer segmentation

## Data

Datasets are not included. CIFAR-10 is downloaded by `torchvision`. The others:

- Animals-10 (Kaggle), expected in `archive/raw-img`
- Census income data (`census.csv`, Udacity version of the UCI Adult dataset)
- Fake and Real News dataset (Kaggle), expected as `data/Fake.csv` and `data/True.csv`
- Student admissions, SMS Spam Collection (UCI), Mall Customers (Kaggle) and small CSV files from the course for the fundamentals

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The notebooks were written with Python 3.10 and the library versions of 2023; newer versions (especially `torchtext`) may need small adjustments.
