import json
import graphviz
import streamlit as st
from streamlit_ace import st_ace
from ontology import RichMLAppProfile

def show_source(value: RichMLAppProfile, height: int = 500):
    code = st_ace(
        value=value.model_dump_json(indent=4),
        language="json",
        theme="monokai",
        height=height,
        font_size=18,
        tab_size=4,
        show_gutter=True,
        wrap = True
    )

    if not code:
        return value

    try:
        data = json.loads(code)
        updated = RichMLAppProfile(**data)
        st.success("Descriptor updated")
        return updated

    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return value

    except Exception as e:
        st.error(f"Schema validation error: {e}")
        return value

def show_diagram(descriptor):
    g = graphviz.Digraph()
    g.attr(rankdir="LR", splines="ortho", bgcolor="transparent")
    g.attr('node', fontname="Inter", fontsize="10", style="filled", shape="box")
    g.attr('edge', color="#58a6ff", penwidth="1.5")

    # Main App node (component with ports)
    g.node("App", f"{descriptor.name}\n(Controller)", fillcolor="#1f6feb", fontcolor="white", shape="component")

    # Observables subgraph - aligned horizontally with App
    with g.subgraph(name="cluster_input") as c:
        c.attr(label="Data Plane", color="#30363d", fontcolor="#8b949e")
        c.attr(rank="same")  # same horizontal rank as App
        for o in descriptor.observables:
            c.node(o.name, f"{o.name}\n({o.frequency})", fillcolor="#238636", fontcolor="white", shape="ellipse")
            g.edge(o.name, "App:4", tailport="e", headport="w")

    # Actions subgraph - Control Plane, aligned horizontally with App
    with g.subgraph(name="cluster_action") as c:
        c.attr(label="Control Plane", color="#30363d", fontcolor="#8b949e")
        c.attr(rank="same")  # keep all actions horizontally aligned with App
        for a in descriptor.actions:
            c.node(a.name, f"{a.name}\nTarget: {a.target}", fillcolor="#da3633", fontcolor="white", shape="box")
            g.edge("App:2", a.name, tailport="e", headport="w")

    # Training Loop node - put outside of horizontal rank subgraphs to stack vertically
    if descriptor.training.required:
        r = descriptor.training.resource
        info = f"TRAINING LOOP\nRes: {r.memory}\nAcc: {r.accelerator}"
        g.node('Train', info, fillcolor='#d29922', fontcolor='black', style='dashed,filled')

        # Connect App bottom port (3) to Train top port (1) vertically
        g.edge('App:3', 'Train:1', style='dashed', dir="both")

        # To keep Train below App and not side-by-side:
        # add invisible edge from Train to one of the control plane nodes or App to enforce layout
        # (optional but recommended)
        if descriptor.actions:
            first_action = descriptor.actions[0].name
            g.edge('Train', first_action, style='invis', constraint='false')

    st.graphviz_chart(g, use_container_width=True)

def show_code(value, lang, height = 500, font = 18, tab = 4):
    return st_ace(value, language=lang, theme="monokai", height=height, 
        font_size=font, tab_size=tab, show_gutter=True, wrap = True)
        
  