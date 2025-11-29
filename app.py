from run import create_model, EMPTY_GENOME, AGENT_PARAMS, AGENT_LOGICS
from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from matplotlib.figure import Figure
import numpy as np
import solara
RED = "#ff0000"
GREEN = "#00ff00"
BLUE = "#0000ff"
ORANGE = "#ffa500"
PURPLE = "#800080"
CYAN = "#00ffff"
MAGENTA = "#ff00ff"
YELLOW = "#ffff00"
BLACK = "#000000"
WHITE = "#ffffff"
GRAY = "#808080"
LIGHT_GRAY = "#d3d3d3"
DARK_GRAY = "#404040"


def hex_to_rgb(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def lerp(a, b, t: float):
    return a + (b - a) * t

def map_color(t: float, color1: str, color2: str) -> str:
    """
    t 在 [0, 1] 之间，把 t 映射到 color1 和 color2 之间的渐变色。
    color1, color2 是 '#RRGGBB' 格式的字符串。
    """
    # 限制在 [0, 1]
    t = max(0.0, min(1.0, t))

    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)

    r = int(lerp(r1, r2, t))
    g = int(lerp(g1, g2, t))
    b = int(lerp(b1, b2, t))

    return rgb_to_hex((r, g, b))


def num_to_color(num: float, low, high, low_c=RED, high_c=GREEN) -> str:
    """
    Map an float value to a color between low_c and high_c. 
    """
    clamped = max(low, min(high, num))
    t = clamped / (high - low)
    return map_color(t, low_c, high_c)


def agent_portrayal(agent):
    return AgentPortrayalStyle(
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        color=num_to_color(agent.sugar, low=0, high=20),
        marker="o",
        size=10,
        zorder=1,
    )


def propertylayer_portrayal(layer):
    if layer.name == "sugar":
        return PropertyLayerStyle(
            color="blue",
            alpha=0.8,
            colorbar=True,
            vmin=0,
            vmax=10,
        )
    # fallback, in case other layers are ever added
    return PropertyLayerStyle(
        color="red",
        alpha=0.8,
        colorbar=True,
        vmin=0,
        vmax=10,
    )


def post_process(ax):
    ax.set_aspect("equal")
    return ax


def make_gene_freq_component(gene_name: str, page: int = 1, alpha: float = 0.6):
    """
    Create a Matplotlib stacked area plot component for the relative
    allele frequencies of a single gene, using data from the model's
    DataCollector.

    The DataCollector is expected to have a model_reporter column
    named f\"{gene_name}_freqs\" that stores a 1D numpy array of
    relative frequencies per time step.

    The `alpha` parameter controls the transparency of the stacked areas.
    """

    def component(model):
        df = model.datacollector.get_model_vars_dataframe()
        col = f"{gene_name}_freqs"

        fig = Figure(figsize=(4, 2.5))
        ax = fig.subplots()

        if col in df.columns and not df.empty:
            # Each entry in the column should be an array of allele frequencies
            data = [np.asarray(v) for v in df[col] if v is not None]

            if data:
                arr = np.vstack(data)  # shape: (T, num_alleles)
                steps = np.arange(arr.shape[0])
                ys = [arr[:, i] for i in range(arr.shape[1])]

                stacks = ax.stackplot(steps, *ys, alpha=alpha)
                ax.set_ylim(0, 1)

                # Legend: one entry per allele
                labels = [f"allele {i + 1}" for i in range(arr.shape[1])]
                ax.legend(stacks, labels, loc="upper right")

        ax.set_title(f"{gene_name} allele frequencies")
        ax.set_xlabel("Step")
        ax.set_ylabel("Relative frequency")

        return solara.FigureMatplotlib(fig)

    return (component, page)


model_params = {
    # Fixed, non-user-adjustable parameters required by Sugarscape.__init__
    "empty_genome": EMPTY_GENOME,
    "agent_params": AGENT_PARAMS,
    "agent_logics": AGENT_LOGICS,
    # Genetic / trait parameters
    "mutation_rate": Slider(
        "Mutation rate", value=0.01, min=0.0, max=0.2, step=0.01
    ),
    "vision_min": Slider("Min Vision", value=1, min=1, max=5, step=1),
    "vision_max": Slider("Max Vision", value=4, min=1, max=8, step=1),
    "metabolism_min": Slider("Min Metabolism", value=1, min=1, max=5, step=1),
    "metabolism_max": Slider("Max Metabolism", value=4, min=1, max=8, step=1),
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": 50,
    "height": 50,
    # Population parameters
    "initial_population": Slider(
        "Initial Population", value=200, min=50, max=500, step=10
    ),
    # Agent endowment parameters
    "endowment_min": Slider("Min Initial Endowment", value=25, min=5, max=30, step=1),
    "endowment_max": Slider("Max Initial Endowment", value=50, min=30, max=100, step=1),
    # # Metabolism parameters
    # "metabolism_min": Slider("Min Metabolism", value=1, min=1, max=3, step=1),
    # "metabolism_max": Slider("Max Metabolism", value=5, min=3, max=8, step=1),
    # # Vision parameters
    # "vision_min": Slider("Min Vision", value=1, min=1, max=3, step=1),
    # "vision_max": Slider("Max Vision", value=5, min=3, max=8, step=1),
}

# instantiate sugar-only model via helper in run.py
model = create_model()

# Space renderer (Matplotlib backend)
renderer = SpaceRenderer(model, backend="matplotlib").render(
    agent_portrayal=agent_portrayal,
    propertylayer_portrayal=propertylayer_portrayal,
    post_process=post_process,
)

# Dashboard
page = SolaraViz(
    model,
    renderer,
    components=[
        # matches model_reporters={"#Agents": ...} in Sugarscape
        make_plot_component("#Agents", page=1),
        *[make_gene_freq_component(gene, page=1, alpha=0.5) for gene in model.gene_names],
    ],
    model_params=model_params,
    name="Sugarscape",
    play_interval=150,
)

page  # noqa
