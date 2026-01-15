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
        font_size=14,
        tab_size=4,
        show_gutter=True,
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
    g.attr('edge', color="#58a6ff", penwidth="1.2")

    # Main App
    g.node("App", f"{descriptor.name}\n(Controller)", fillcolor="#1f6feb", fontcolor="white", shape="component")

    # Observables
    with g.subgraph(name="cluster_input") as c:
        c.attr(label="Data Plane", color="#30363d", fontcolor="#8b949e")
        for o in descriptor.observables:
            c.node(o.name, f"{o.name}\n({o.frequency})", fillcolor="#238636", fontcolor="white", shape="ellipse")
            g.edge(o.name, "App")

    # Actions
    with g.subgraph(name="cluster_action") as c:
        c.attr(label="Control Plane", color="#30363d", fontcolor="#8b949e")
        for a in descriptor.actions:
            c.node(a.name, f"{a.name}\nTarget: {a.target}", fillcolor="#da3633", fontcolor="white", shape="box")
            g.edge("App", a.name)

    # Resources
    if descriptor.training.required:
        r = descriptor.training.resource
        info = f"TRAINING LOOP\nRes: {r.memory}\nAcc: {r.accelerator}"
        g.node('Train', info, fillcolor='#d29922', fontcolor='black', style='dashed,filled')
        g.edge('App', 'Train', style='dashed', dir="both")

    st.graphviz_chart(g, use_container_width=True)

def show_code(value, lang, height = 500):
    return st_ace(value, language=lang, theme="monokai", height=height, 
        font_size=14, tab_size=4, show_gutter=True)
        
  