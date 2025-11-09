import numpy as np
import plotly.graph_objects as go
import torch


def visualize_layer_results_core(
    probs: torch.Tensor,
    inputs: list[str],
    decoded: list[list[list[str]]],
    y_labels: list[str] | None = None,
):
    # Assume probs is of shape (LAYER, SEQUENCE, N)
    layers, sequences, top_k = probs.shape
    assert len(inputs) == sequences, f"{len(inputs)} != {sequences}"
    assert len(decoded) == layers, f"{len(decoded)} != {layers}"
    assert all(len(decoded[layer]) == sequences for layer in range(layers)), (
        f"{decoded[0]} !?= {sequences}"
    )
    assert all(
        len(decoded[layer][position]) == top_k
        for layer in range(layers)
        for position in range(sequences)
    )

    # We will create a heatmap for each layer showing top-k logits
    fig = go.Figure()

    # top1 data
    z_data = probs[:, :, 0]
    x_labels = inputs
    if y_labels is None:
        y_labels = np.arange(layers)
    text = [[position[0] for position in layer] for layer in decoded]

    # topk text data
    hover_text = [
        [
            "<br>".join(
                [
                    f"{probs[layer_idx][position][k]:.2f}:{decoded[layer_idx][position][k]}"
                    for k in range(top_k)
                ]
            )
            for position in range(sequences)
        ]
        for layer_idx in range(layers)
    ]

    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertext=hover_text,
            hovertemplate="<b>Input:</b> %{x}<br><b>Layer:</b> %{y}<br><br>%{hovertext}",
            name="",
        )
    )

    # Update layout to make the plot more readable
    fig.update_layout(
        title="Logit Lens Visualization",
        xaxis_title="Sequence",
        yaxis_title="Layer",
        height=30 * layers,
        showlegend=False,
    )

    # Show the figure
    return fig
