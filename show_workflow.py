from IPython.display import Image, display
from workflow.graph import graph

from workflow.graph import graph

png_data = graph.get_graph().draw_mermaid_png()

with open("workflow.png", "wb") as file:
    file.write(png_data)