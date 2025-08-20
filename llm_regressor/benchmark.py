import os
import json
import random
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from llm_regressor import DATASET_NAME
from llm_regressor.evaluation import Tester
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from llm_regressor.plotter import plot_benchmark_results_historgram

random.seed(42)
_w2v_cache = {}


def get_w2v_features(train):
    """
    Returns X_w2v, y_prices.
    Uses internal cache to avoid recomputation.
    """
    key = id(train)  # or some hash if train can change
    if key in _w2v_cache:
        return _w2v_cache[key]

    documents = [description(item) for item in train]
    prices = np.array([float(item["price"]) for item in train])

    processed_docs = [simple_preprocess(doc) for doc in documents]
    w2v_model = Word2Vec(
        sentences=processed_docs,
        vector_size=400,
        window=5,
        min_count=1,
        workers=8,
    )

    def document_vector(doc):
        words = simple_preprocess(doc)
        vecs = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        return (
            np.mean(vecs, axis=0) if vecs else np.zeros(w2v_model.vector_size)
        )

    X_w2v = np.array([document_vector(doc) for doc in documents])

    _w2v_cache[key] = (
        X_w2v,
        prices,
        document_vector,
    )  # save model too if needed
    return X_w2v, prices, document_vector


def benchmark_wrapper(name):
    """
    Decorator that takes care of skipping, running, testing, and saving results.
    Benchmark function should return a prediction function that takes an item and returns a price.
    """

    def decorator(fn):
        def wrapper(benchmark_results, train, test):
            if name in benchmark_results:
                print(f"Skipping {name}, already exists in results.")
                return benchmark_results

            print(f"Running benchmark: {name}")
            predict_fn = fn(train)  # Train and return prediction function
            tester = Tester(predict_fn, test, title=name)
            tester.run()
            benchmark_results[name] = tester.to_dict()

            with open("benchmark_results.json", "w") as f:
                json.dump(benchmark_results, f, indent=2)

            return benchmark_results

        return wrapper

    return decorator


def description(item):
    text = item["text"].replace(
        "How much does this cost to the nearest dollar?\n\n", ""
    )
    return text.split("\n\nPrice is $")[0]


def random_pricer(item):
    return random.randrange(1, 1000)


class ConstantPricer:
    def __init__(self):
        self.training_average = None

    def train(self, dataset):
        training_prices = [item["price"] for item in dataset]
        self.training_average = sum(training_prices) / len(training_prices)

    def predict(self, item):
        if self.training_average is None:
            raise ValueError("Model has not been trained yet.")
        return self.training_average


"""
Benchmark functions for different models with decorators applied.
"""


@benchmark_wrapper("Random Pricer")
def random_pricer_fn(train):
    return random_pricer


@benchmark_wrapper("Constant Pricer")
def constant_pricer_fn(train):
    constant_pricer = ConstantPricer()
    constant_pricer.train(train)
    return constant_pricer.predict


@benchmark_wrapper("BoW LR Pricer")
def bow_lr_pricer_fn(train):
    documents = [description(item) for item in train]
    prices = np.array([float(item["price"]) for item in train])

    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(documents)
    regressor = LinearRegression().fit(X, prices)

    def bow_lr_pricer(item):
        return max(regressor.predict(vectorizer.transform([item]))[0], 0)

    return bow_lr_pricer


@benchmark_wrapper("Word2Vec LR Pricer")
def word2vec_lr_pricer_fn(train):
    from sklearn.linear_model import LinearRegression

    X_w2v, prices, document_vector = get_w2v_features(train)
    regressor = LinearRegression().fit(X_w2v, prices)

    def predict(item):
        return max(0, regressor.predict([document_vector(item)])[0])

    return predict


@benchmark_wrapper("Word2Vec LSVR Pricer")
def word2vec_lsvr_pricer_fn(train):
    from sklearn.svm import LinearSVR

    X_w2v, prices, document_vector = get_w2v_features(train)
    regressor = LinearSVR().fit(X_w2v, prices)

    def predict(item):
        return max(0, regressor.predict([document_vector(item)])[0])

    return predict


@benchmark_wrapper("Word2Vec Random Forest Pricer")
def word2vec_rf_pricer_fn(train):
    from sklearn.ensemble import RandomForestRegressor

    X_w2v, prices, document_vector = get_w2v_features(train)
    rf_model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=8
    )
    rf_model.fit(X_w2v, prices)

    def predict(item):
        return max(0, rf_model.predict([document_vector(item)])[0])

    return predict


@benchmark_wrapper("GPT-4-Mini")
def gpt4_mini_pricer_fn(train):
    from llm_regressor.rag_regressor.gpt_rag_mini import GPT4MiniRAG

    model = GPT4MiniRAG()
    return model.price


@benchmark_wrapper("GPT-4-Mini with RAG")
def gpt4_mini_rag_pricer_fn(train):
    import chromadb
    from llm_regressor.rag_regressor.gpt_rag_mini import GPT4MiniRAG

    DB = "products_vectorstore"
    # Connect to the Chroma datastore
    client = chromadb.PersistentClient(path=DB)
    collection = client.get_or_create_collection(name="products")

    model = GPT4MiniRAG(collection=collection)
    return model.price


def run_benchmark():
    # Load existing benchmark results if available
    if os.path.exists("benchmark_results.json"):
        with open("benchmark_results.json", "r") as f:
            benchmark_results = json.load(f)
            print("Loaded existing results.")
    else:
        benchmark_results = {}

    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]
    test = dataset["test"]

    # Run all benchmarks models
    benchmark_results = random_pricer_fn(benchmark_results, train, test)
    benchmark_results = constant_pricer_fn(benchmark_results, train, test)
    benchmark_results = bow_lr_pricer_fn(benchmark_results, train, test)
    benchmark_results = word2vec_lr_pricer_fn(benchmark_results, train, test)
    benchmark_results = word2vec_lsvr_pricer_fn(benchmark_results, train, test)
    # benchmark_results = word2vec_rf_pricer_fn(benchmark_results, train, test)
    benchmark_results = gpt4_mini_pricer_fn(benchmark_results, train, test)
    benchmark_results = gpt4_mini_rag_pricer_fn(benchmark_results, train, test)

    return benchmark_results


if __name__ == "__main__":
    benchmark_results = run_benchmark()
    plot_benchmark_results_historgram(benchmark_results)
