## Overview

This project is based on Mesa’s `SugarscapeG1mt` example with explicit genetics and richer agent life‑history, to explore questions in evolutionary psychology and dual‑inheritance theory.

The key idea is that “learning capacity” and behavioural flexibility are themselves heritable traits: genes shape not only what agents do, but how easily they can adjust their strategies in changing environments.

## What’s in the model

- Sugarscape‑style resource world: agents move on a 2D grid, harvest sugar, metabolize it, and die if they run out.
- Genetic architecture: genes, loci, karyotypes, mutation, gamete formation, recombination; allele frequencies are tracked over time.
- Trait mapping: vision, metabolism, and life‑history parameters are derived from genes.
- Reproduction: age, spatial constraints, cooldowns, maximum sugar, maximum children, and maximum age, plus parental sugar transfer to offspring.
- Visualization: a Solara dashboard with a Matplotlib space view, time series of population size, stacked allele‑frequency plots, and sliders for ecological, genetic, and life‑history parameters.