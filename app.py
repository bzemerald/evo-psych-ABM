from pathlib import Path

from model import Sugarscape
from agents import AgentParams, DefaultAgentLogics, SugarscapeAgent
from genetic import EmptyLociCollection, Gene
from mapmaker import *
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


class ActivityGene(Gene):
    """conbining vison and metabolism"""

class ControlGene(Gene):
    """Neutral control gene (no direct effect used in logic)."""

class BasicAgentLogics(DefaultAgentLogics):
    """
    Agent logics that derive metabolism and vision from genotype
    using MetabolismGene and VisionGene.
    """

    @staticmethod
    def _allele_to_int(
        allele_index: int, min_val: int, max_val: int, num_alleles: int
    ) -> int:
        """Map an allele index (1..num_alleles) to an integer in [min_val, max_val]."""
        if num_alleles <= 1:
            return int(min_val)
        frac = (allele_index - 1) / (num_alleles - 1)
        return int(round(min_val + frac * (max_val - min_val)))

    def _gene_trait(self, agent, gene_name: str, min_val: int, max_val: int) -> int:
        """Compute an integer trait value from the given gene using the agent's genotype."""
        genes = agent.genotype.by_name.get(gene_name, [])
        if not genes:
            return int(round((min_val + max_val) / 2))
        num_alleles = type(genes[0]).num_alleles
        values = [
            self._allele_to_int(g.allele, min_val, max_val, num_alleles) for g in genes
        ]
        trait = int(round(sum(values) / len(values)))
        return max(min_val, min(max_val, trait))

    def metabolism(self, agent) -> float:
        m_min = int(agent.model.metabolism_min)
        m_max = int(agent.model.metabolism_max)
        return self._gene_trait(agent, "MetabolismGene", m_min, m_max)

    def vision(self, agent) -> int:
        v_min = int(agent.model.vision_min)
        v_max = int(agent.model.vision_max)
        return self._gene_trait(agent, "VisionGene", v_min, v_max)

    def can_breed(self, agent: SugarscapeAgent) -> bool:
        return super().can_breed(agent)


class StrategicAgentLogics(BasicAgentLogics):
    def genetic_strategy(self, agent: SugarscapeAgent) -> float:
        return agent.genotype.phenotype("ActivityGene")

    def strategy(self, agent: SugarscapeAgent) -> float:
        """
        A float between 0 to 1 to denote the foraging strategy.
        0 -> high vision, low metabolism
        1 -> low vision, high metabolism
        """
        return self.genetic_strategy(agent)

    def metabolism(self, agent: SugarscapeAgent) -> float:
        minm, maxm = agent.params.min_metabolism, agent.params.max_metabolism
        return lerp(minm, maxm, self.genetic_strategy(agent))

    def vision(self, agent: SugarscapeAgent) -> int:
        minv, maxv = agent.params.min_vision, agent.params.max_vision
        return round(lerp(minv, maxv, self.genetic_strategy(agent)))




EMPTY_GENOME = EmptyLociCollection(
    ["ControlGene", "ActivityGene"]
)

INI_MAP = np.genfromtxt(Path(__file__).parent / "sugarmaps/noise.txt")

AGENT_PARAMS = AgentParams()

AGENT_LOGICS = StrategicAgentLogics()


def hex_to_rgb(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def lerp(a, b, t: float, inv_gamma=1):
    rev = 1/inv_gamma 
    a, b = a**rev, b**rev
    return (a + (b - a) * t)**inv_gamma

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
        color=num_to_color(agent.logics.strategy(agent), low=0, high=1, low_c=GREEN, high_c=ORANGE),
        marker="o",
        size=lerp(0,15, agent.sugar / agent.params.min_reproduction_sugar),
        zorder=1,
    )


def propertylayer_portrayal(layer):
    if layer.name == "sugar":
        return PropertyLayerStyle(
            color="blue",
            alpha=0.6,
            colorbar=True,
            vmin=0,
            vmax=10,
        )


def post_process(chart):
    # Adjust overall size for the Altair space visualization
    return chart.properties(width=600, height=600)


def limit_0_1(ax):
    ax.set_ylim(0, 1)
    return ax

def limit_0(ax):
    ax.set_ylim(0)
    return ax

def make_gene_freq_component(gene_name: str, page: int = 1, alpha: float = 0.6):
    """
    Create a Matplotlib stacked area plot component for the relative allele
    frequencies of a single gene, using data from the model's DataCollector.

    The DataCollector is expected to have a model_reporter column
    named f\"{gene_name}_freqs\" that stores a 1D numpy array of
    relative frequencies per time step.

    The `alpha` parameter controls the transparency of the stacked areas.
    """

    def component(model):
        df = model.datacollector.get_model_vars_dataframe()
        col = f"{gene_name}_freqs"

        # make some extra space on the right for the legend
        fig = Figure(figsize=(5, 3))
        ax = fig.subplots()
        fig.subplots_adjust(right=0.75, bottom=0.2, top=0.8)

        if col in df.columns and not df.empty:
            # Each entry in the column should be an array of allele frequencies
            data = [np.asarray(v) for v in df[col] if v is not None]

            if data:
                arr = np.vstack(data)  # shape: (T, num_alleles)
                steps = np.arange(arr.shape[0])
                num_alleles = arr.shape[1]
                ys = [arr[:, i] for i in range(num_alleles)]

                line_handle = None

                # Use a smooth color gradient across alleles
                if num_alleles > 1:
                    colors = [
                        map_color(i / (num_alleles - 1), GREEN, ORANGE)
                        for i in range(num_alleles)
                    ]
                else:
                    colors = [map_color(0.5, GREEN, ORANGE)]

                stacks = ax.stackplot(steps, *ys, alpha=alpha, colors=colors)
                ax.set_ylim(0, 1)

                # Compute and plot average effect size as a red line
                gene_cls = Gene.loci_registry.get(gene_name)
                if gene_cls is not None:
                    effects = np.asarray(gene_cls.effect_array(), dtype=float)
                    if effects.shape[0] >= num_alleles:
                        avg_effect = (arr * effects[:num_alleles]).sum(axis=1)
                        line_handle = ax.plot(
                            steps,
                            avg_effect,
                            color="red",
                            linewidth=2,
                            label="avg effect size",
                        )[0]

                # Legend: one entry per allele, plus average effect size, outside the axes
                labels = [f"allele {i + 1}" for i in range(arr.shape[1])]
                handles = list(stacks)
                if line_handle is not None:
                    handles.append(line_handle)
                    labels.append("avg effect size")

                ax.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0,
                )

        ax.set_title(f"{gene_name} allele frequencies")
        ax.set_xlabel("Step")
        ax.set_ylabel("Relative frequency")

        return solara.FigureMatplotlib(fig)

    return (component, page)


model_params = {
    # Fixed, non-user-adjustable parameters required by Sugarscape.__init__
    "empty_genome": EMPTY_GENOME,
    "agent_logics": AGENT_LOGICS,

    "society_type": Slider(
        "Society Type", value=0.5, min=0, max=1, step=0.01
    ),
    "resource_richness":Slider(
        "Resource Regen", value=0.5, min=0, max=1, step=0.01
    ),
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    # Genetic / trait parameters
    "mutation_rate": Slider(
        "Mutation rate", value=0.05, min=0.0, max=0.2, step=0.01
    ),
    "num_alleles": 8,
    "vision_min": Slider("Min Vision", value=2, min=1, max=10, step=1),
    "vision_max": Slider("Max Vision", value=10, min=1, max=20, step=1),
    "metabolism_min": Slider("Min Metabolism", value=2, min=1, max=5, step=0.1),
    "metabolism_max": Slider("Max Metabolism", value=3.5, min=1, max=8, step=0.1),
    # Population parameters
    "min_reproduction_sugar": 30,
    "initial_population": 100,
    # Agent endowment parameters
    "endowment": 20,
    # Grid parameters
    "grid_size": 70,
    # Agent parameters (map directly to Sugarscape.__init__)
    "initial_sugar": 0,
    "reproduction_age": 1,
    "reproduction_cooldown": 1,
    "max_children": float('inf'),
    "max_age": float('inf')
}

def _defaults_from_model_params(params: dict) -> dict:
    """
    Extract default keyword arguments for Sugarscape from model_params.
    - Slider: use its .value
    - dict with \"value\" (e.g. InputText spec): use that value
    - any other object: use as-is (e.g. EMPTY_GENOME, AGENT_LOGICS)
    """
    kwargs: dict = {}
    for name, opt in params.items():
        if isinstance(opt, Slider):
            kwargs[name] = opt.value
        elif isinstance(opt, dict) and "value" in opt:
            kwargs[name] = opt["value"]
        else:
            kwargs[name] = opt
    return kwargs


# Instantiate initial model directly from model_params defaults
_initial_kwargs = _defaults_from_model_params(model_params)
model = Sugarscape(**_initial_kwargs)

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
        make_plot_component("#Agents", post_process=limit_0,page=1),
        *[make_gene_freq_component(gene, page=0, alpha=0.8) for gene in model.gene_names],
    ], # type: ignore
    model_params=model_params,
    name="Sugarscape",
    play_interval=150,
)

page # type: ignore
