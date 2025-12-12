import plotly.graph_objects as go
import polars as pl


def muscle_length_parcoords(
    name: str, data: pl.DataFrame, length_col: str
) -> go.Figure:
    """
    Create a parallel coordinates plot for muscle lengths and the coordinates.
    """

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=data[length_col], colorscale="Viridis", showscale=True),
            dimensions=[dict(label=col, values=data[col]) for col in data.columns],
        )
    )
    fig.update_layout(title=f"{name} Lengths")
    return fig
