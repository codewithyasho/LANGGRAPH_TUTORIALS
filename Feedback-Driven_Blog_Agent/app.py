import streamlit as st
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from typing import TypedDict
from pydantic import BaseModel, Field

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Feedback-Driven Blog Agent",
    page_icon="✍️",
    layout="wide"
)

# ---------------- STATE ----------------


class BlogState(TypedDict):
    topic: str
    outline: str
    final_blog: str
    blog_score: float
    feedback: str


# ---------------- STRUCTURED OUTPUT ----------------

class BlogEvaluation(BaseModel):
    score: float = Field(..., ge=0.0, le=10.0)
    feedback: str = Field(...,
                          description="What needs improvement in the blog")


# ---------------- MODELS ----------------

@st.cache_resource
def get_models():
    llm = ChatOllama(model="deepseek-v3.1:671b-cloud", temperature=0.7)
    llm_judge = ChatOllama(model="deepseek-v3.1:671b-cloud", temperature=0.0)
    return llm, llm_judge


llm, llm_judge = get_models()


# ---------------- NODES ----------------

def create_outline(state: BlogState) -> BlogState:
    prompt = f"Create a detailed blog outline on: {state['topic']}"
    state["outline"] = llm.invoke(prompt).content
    return state


def create_blog(state: BlogState) -> BlogState:
    outline = state["outline"]
    feedback = state.get("feedback", "")

    if feedback:
        prompt = f"""
Improve the blog using this feedback:

Feedback:
{feedback}

Blog outline:
{outline}

Rewrite the blog better.
"""
    else:
        prompt = f"""
Write a detailed blog based on this outline:

{outline}
"""

    state["final_blog"] = llm.invoke(prompt).content
    return state


def score_blog(state: BlogState) -> BlogState:
    prompt = f"""
Evaluate the blog based on how well it follows the outline.

Outline:
{state['outline']}

Blog:
{state['final_blog']}

Return JSON only:
{{"score": number between 0 and 10, "feedback": "how to improve"}}
"""

    judge = llm_judge.with_structured_output(BlogEvaluation)
    result = judge.invoke(prompt)

    state["blog_score"] = result.score
    state["feedback"] = result.feedback

    return state


# ---------------- ROUTER ----------------

def blog_optimizer(state: BlogState) -> str:
    if state["blog_score"] < 7.0:
        return "optimize blog"
    return "end"


# ---------------- GRAPH ----------------

@st.cache_resource
def build_graph():
    graph = StateGraph(BlogState)

    graph.add_node("outline_node", create_outline)
    graph.add_node("blog_node", create_blog)
    graph.add_node("score_node", score_blog)

    graph.add_edge(START, "outline_node")
    graph.add_edge("outline_node", "blog_node")
    graph.add_edge("blog_node", "score_node")

    graph.add_conditional_edges(
        source="score_node",
        path=blog_optimizer,
        path_map={
            "optimize blog": "blog_node",
            "end": END
        }
    )

    return graph.compile()


app = build_graph()


# ---------------- STREAMLIT UI ----------------

st.title("✍️ Feedback-Driven Blog Agent")
st.markdown("### AI-powered blog generator with automated quality feedback loop")

st.divider()

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Using **DeepSeek-V3.1** model")
    st.markdown("""
    **How it works:**
    1. 📝 Creates an outline
    2. ✍️ Writes the blog
    3. 🔍 Scores the quality (0-10)
    4. 🔄 Refines if score < 7.0
    5. ✅ Returns final blog
    """)

    st.divider()
    st.caption("Powered by LangGraph + Ollama")

# Main input
col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "Enter your blog topic:",
        placeholder="e.g., Cats vs Dogs: Which Pet Is Better?",
        key="topic_input"
    )

with col2:
    st.write("")  # Spacer
    st.write("")  # Spacer
    generate_button = st.button(
        "🚀 Generate Blog", type="primary", use_container_width=True)

st.divider()

# Generate blog on button click
if generate_button and topic:
    with st.spinner("🤖 AI is working on your blog..."):

        # Progress tracking
        progress_placeholder = st.empty()
        outline_placeholder = st.empty()
        iterations_placeholder = st.empty()

        try:
            # Stream events for real-time updates
            iteration_count = 0

            with st.status("Processing...", expanded=True) as status:
                st.write("📝 Creating outline...")

                response = app.invoke({"topic": topic})

                status.update(label="✅ Blog generated successfully!",
                              state="complete", expanded=False)

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📄 Final Blog", "📋 Outline", "📊 Quality Report", "🔄 Process"])

            with tab1:
                st.markdown("### Final Blog")
                st.markdown("---")
                st.markdown(response["final_blog"])

                # Download button
                st.download_button(
                    label="📥 Download Blog",
                    data=response["final_blog"],
                    file_name=f"{topic[:30]}.txt",
                    mime="text/plain"
                )

            with tab2:
                st.markdown("### Blog Outline")
                st.markdown("---")
                st.markdown(response["outline"])

            with tab3:
                st.markdown("### Quality Assessment")

                col1, col2 = st.columns(2)

                with col1:
                    score = response["blog_score"]
                    st.metric(
                        label="Final Score",
                        value=f"{score}/10",
                        delta="Excellent" if score >= 8 else "Good" if score >= 7 else "Needs Improvement"
                    )

                with col2:
                    # Score indicator
                    if score >= 8:
                        st.success("🌟 High Quality Blog")
                    elif score >= 7:
                        st.info("✅ Good Quality Blog")
                    else:
                        st.warning("⚠️ Acceptable Quality")

                st.markdown("---")
                st.markdown("### 📝 Feedback")
                st.info(response["feedback"])

            with tab4:
                st.markdown("### Process Information")
                st.json({
                    "topic": response["topic"],
                    "final_score": response["blog_score"],
                    "workflow": "outline → blog → score → conditional refinement loop"
                })

                st.markdown("### Graph Visualization")
                st.markdown("""
                ```
                START → Outline Node → Blog Node → Score Node
                                         ↑            ↓
                                         └────(if score < 7)
                                                   ↓
                                                  END (if score ≥ 7)
                ```
                """)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

elif generate_button and not topic:
    st.warning("⚠️ Please enter a blog topic first!")
