from model import Sugarscape
from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle


def agent_portrayal(agent):
    return AgentPortrayalStyle(
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        color="red",
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


def post_process(chart):
    chart = chart.properties(width=400, height=400)
    return chart


model_params = {
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
    # Metabolism parameters
    "metabolism_min": Slider("Min Metabolism", value=1, min=1, max=3, step=1),
    "metabolism_max": Slider("Max Metabolism", value=5, min=3, max=8, step=1),
    # Vision parameters
    "vision_min": Slider("Min Vision", value=1, min=1, max=3, step=1),
    "vision_max": Slider("Max Vision", value=5, min=3, max=8, step=1),
    # no trade toggle here – sugar-only model
}

# instantiate sugar-only model
model = Sugarscape()

# Space renderer (Altair backend)
renderer = SpaceRenderer(model, backend="altair").render(
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
    ],
    model_params=model_params,
    name="Sugarscape (Sugar Only)",
    play_interval=150,
)

page  # noqa