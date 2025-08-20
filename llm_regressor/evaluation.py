import math
import numpy as np
import plotly.graph_objects as go
from llm_regressor import COLOR_MAP, RESET


class Tester:
    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []
        self.fig = None

    def to_dict(self):
        def to_builtin(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        return {
            "title": self.title,
            "guesses": [to_builtin(x) for x in self.guesses],
            "truths": [to_builtin(x) for x in self.truths],
            "errors": [to_builtin(x) for x in self.errors],
            "sles": [to_builtin(x) for x in self.sles],
            "colors": self.colors,
        }

    def color_for(self, error, truth):
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint["text"])
        truth = datapoint["price"]
        error = abs(guess - truth)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error**2
        color = self.color_for(error, truth)
        # title = datapoint["text"].split("\n\n")[1][:20] + "..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        # print(
        #     f"{COLOR_MAP[color]}{i + 1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}"
        # )

    def print_details(self):
        for color, i, guess, truth, error, sle, title in zip(
            self.colors,
            range(self.size),
            self.guesses,
            self.truths,
            self.errors,
            self.sles,
            [d["text"].split("\n\n")[1][:20] + "..." for d in self.data],
        ):
            print(
                f"{COLOR_MAP[color]}{i + 1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}"
            )

    def chart(self, title):
        max_val = max(max(self.truths), max(self.guesses))
        fig = go.Figure()
        # Add diagonal (perfect prediction line)
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="deepskyblue", width=2, dash="dot"),
                name="Ideal",
            )
        )
        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=self.truths,
                y=self.guesses,
                mode="markers",
                marker=dict(color=self.colors, size=6, opacity=0.7),
                name="Predictions",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Ground Truth",
            yaxis_title="Model Estimate",
            width=900,
            height=700,
            template="plotly_white",
        )
        # Save figure in the object
        self.fig = fig

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color == "green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits / self.size * 100:.1f}%"
        self.chart(title)

    def run(self):
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()


if __name__ == "__main__":
    from datasets import load_dataset
    from llm_regressor.model import Llama3Model
    from llm_regressor import DATASET_NAME

    # Get model
    model = Llama3Model()
    # Get dataset
    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]
    test = dataset["test"]
    # Predict and evaluate
    model.predict(test[0]["text"])
    Tester.test(model.predict, test)
