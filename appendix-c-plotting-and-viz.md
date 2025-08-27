---
date: 2025-08-21
---

# Appendix C: Python Plotting & Chemical Structure Visualization

> **One-sentence takeaway.** Python's visualization ecosystem spans general-purpose plotting and domain-specific 3D chemical viewing; choose tools based on interactivity, dimensionality (2D/3D), and data modality.

---

## General plotting libraries (Python)

Use this quick catalog to select a plotting stack based on needs such as interactivity, grammar-of-graphics, or 3D/volumetric data.

- **[Matplotlib](https://matplotlib.org/)**: Foundational 2D plotting library for static, animated, and interactive figures; fine control for publication formatting; base for many higher-level APIs.
- **[Seaborn](https://seaborn.pydata.org/)**: High-level statistical visualization on top of Matplotlib with consistent themes and concise APIs for distributions, regression, and categorical plots.
- **[Plotly](https://plotly.com/) ([Plotly.py](https://plotly.com/python/))**: Interactive 2D/3D graphing in notebooks and the web (scatter, contour, surface, maps); exportable to static images.
- **[Bokeh](https://bokeh.org/)**: Interactive browser-based plotting with a Python server for streaming data and linked brushing.
- **[Altair](https://altair-viz.github.io/)**: Declarative "grammar of graphics" backed by Vega-Lite; great for tidy (long-form) data with automatic interactivity.
- **[plotnine](https://plotnine.org/)**: A Python implementation of ggplot2's grammar of graphics; static output with concise layered syntax.
- **[Pandas plotting](https://pandas.pydata.org/docs/user_guide/visualization.html) API**: `DataFrame.plot(...)` convenience wrappers over Matplotlib for quick exploratory analysis.
- **[pygal](https://www.pygal.org/en/stable/)**: Lightweight SVG charts; ideal for static sites where crisp vector output is preferred.
- **[missingno](https://github.com/ResidentMario/missingno)**: Visual diagnostics for missing-data patterns (nullity matrix/heatmap/dendrogram).
- **[Leather](https://leather.readthedocs.io/en/latest/)**: Minimal charting that favors correctness and SVG output over exhaustive features.
- **[Mayavi](https://docs.enthought.com/mayavi/mayavi/)**: 3D scientific visualization (VTK-based) for isosurfaces, volume rendering, and vector fields.
- **[NetworkX](https://networkx.org/)**: Graph creation/analysis with 2D/3D drawing (via Matplotlib, Graphviz, or VTK backends).
- **[python-igraph](https://python.igraph.org/en/stable/)**: High-performance network analysis with robust layouts and scalable plotting.
- **[pymatviz](https://pymatviz.janosh.dev/)**: "A toolkit for visualizations in materials informatics."

### Pick-by-task (plotting)

| Use case | Recommended libraries |
|---|---|
| General static plots | Matplotlib, Seaborn, Pandas API |
| Interactive web plots | Bokeh, Plotly, Altair |
| Grammar-of-graphics | Altair, plotnine |
| SVG/lightweight charts | pygal, Leather |
| Missing-data diagnostics | missingno |
| 3D/volumetric scientific | Mayavi |
| Networks/graphs | NetworkX, python-igraph |
| Materials visualization | pymatviz |

---

## Chemical & crystallographic visualization (Python)

Tools below span high-quality rendering, interactive model building, and crystallographic data workflows.

- **[Beautiful Atoms](https://beautiful-atoms.readthedocs.io/en/latest/)**: Python control over Blender for high-quality static/animated renders of atomistic structures.
- **[PyMOL](https://www.pymol.org/)**: Mature, scriptable 3D molecular viewer (ball-and-stick, ribbons, surfaces) suitable for publication figures.
- **[Jmol/JSmol](https://jmol.sourceforge.net/)**: Scriptable Java/Web viewer (via JSmol) for interactive 3D structures; embeddable in teaching pages.
- **[NGLView](https://nglviewer.org/nglview/latest/)**: Jupyter widget for interactive 3D molecular visualization; integrates with MDAnalysis and Py3Dmol.
- **[3Dmol.js](https://3dmol.csb.pitt.edu/)**: WebGL-based interactive 3D viewer embeddable in notebooks and web pages; supports various molecular formats.
- **[ASE](https://ase-lib.org/ase/visualize/visualize.html#visualization)**: Atomic Simulation Environment, a set of tools for setting up, running, and analyzing atomistic simulations.
- **[Ovito](https://www.ovito.org/docs/current/python/)**: Open-source software for visualizing and analyzing atomistic simulation data.
- **[MatterViz](https://matterviz.janosh.dev/)**: "A toolkit for building interactive web UIs for materials science."

### Pick-by-task (chemical/crystallographic)

| Goal | Recommended tools |
|---|---|
| High-quality static/animated renders | Beautiful Atoms (Blender), Ovito, PyMOL |
| Interactive 3D viewing (small molecules, crystals) | Jmol/JSmol, NGLView, 3Dmol.js, ASE, MatterViz |
| Web embeds for teaching materials | Jmol/JSmol |

---

### References & further reading

#### Plotting

- The 7 most popular ways to plot data in Python: <https://opensource.com/article/20/4/plot-data-python>
- A curated list of awesome data visualization libraries and tools: <https://github.com/hal9ai/awesome-dataviz>
- Plotting in Python: Comparing the Options: <https://anvil.works/blog/plotting-in-python>
- Python Graph Gallery: <https://python-graph-gallery.com/>

#### Chemical & crystallographic visualization

- What are the freely available crystal-structure visualization ...: <https://mattermodeling.stackexchange.com/questions/467/what-are-the-freely-available-crystal-structure-visualization-softwares>
- Python for Crystallographic Computing: <https://www.numberanalytics.com/blog/crystallographic-computing-python>
