import plotly.graph_objects as go

# Define colors for categories
category_colors = {
    "Regressor": "skyblue",
    "RAG": "blueviolet",
    "LLM": "steelblue",
    "Other": "lightgray",
}

model_categories = {
    "BoW LR Pricer": "Regressor",
    "Word2Vec LR Pricer": "Regressor",
    "Word2Vec LSVR Pricer": "Regressor",
    "Word2Vec Random Forest Pricer": "Regressor",
    "GPT-4-Mini": "RAG",
    "GPT-4-Mini with RAG": "RAG",
    "Llama-3.1-8B": "LLM",
    "Llama-3.1-8B (Fine-tuned)": "LLM",
}


def plot_benchmark_results_historgram(benchmark_results):
    # Compute average errors
    average_errors = {}
    colors = []
    for model, data in benchmark_results.items():
        avg_error = sum(data["errors"]) / len(data["errors"])
        average_errors[model] = avg_error
        colors.append(category_colors[model_categories.get(model, "Other")])

    # Create a bar chart (histogram style)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(average_errors.keys()),
            y=list(average_errors.values()),
            text=[f"{v:.2f}" for v in average_errors.values()],
            textposition="auto",
            marker_color=colors,
        )
    )

    fig.update_layout(
        title="Average Absolute Error per Model",
        xaxis_title="Model",
        yaxis_title="Average Absolute Error",
    )
    fig.show(renderer="browser")
