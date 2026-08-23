import streamlit as st
from rag import config, rag_pipeline, vectorstore
from rag.memory import ConversationMemory
st.set_page_config(page_title="CineRAG", page_icon="🎬", layout="wide")
@st.cache_resource
def _startup_check():
    """Runs once per process, not per request. Fails loudly and clearly if
    secrets are missing, instead of crashing three calls deep with a raw
    traceback."""
    try:
        config.validate()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    vectorstore.ensure_index_loaded()
    return True
def _render_candidates(candidates: list[dict]) -> None:
    """Shared card renderer for both replayed history and a fresh response,
    so the two don't drift out of sync with each other."""
    cols = st.columns(min(len(candidates), 5))
    for col, c in zip(cols, candidates[:5]):
        with col:
            if c.get("poster_url"):
                st.image(c["poster_url"], use_container_width=True)
            # `.get('live_vote_average', fallback)` is wrong here: OMDb sets
            # the key to None (not missing) when it has no rating for a
            # title, and .get()'s default only applies to a MISSING key —
            # so a None value would render as the literal text "None"
            # instead of falling back to the catalog rating. `or` correctly
            # treats None (and 0) as "use the fallback instead."
            rating = c.get("live_vote_average") or c.get("vote_average")
            badge = " 🏆 Top Pick" if c.get("is_top_pick") else ""
            st.caption(f"**{c['title']}**{badge}  \n⭐ {rating}")
_startup_check()
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # [(role, text, candidates)]
st.title("🎬 CineRAG")
st.caption(f"LLM backend: {config.LLM_BACKEND} · ask for recommendations, then refine them")

st.markdown(
    """
    <div style="
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
        font-size: 12px;
        color: #8a8a8a;
        z-index: 999999;
    ">
        CineRAG · by <b style="color:#b5b5b5;">Dhanush</b> · powered by <b style="color:#b5b5b5;">Groq</b>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Filters")
    language = st.selectbox(
        "Language",
        options=[None, "en", "hi", "ko", "fr", "ja", "es"],
        format_func=lambda x: "Any" if x is None else x,
    )
    if st.button("Clear conversation"):
        st.session_state.memory = ConversationMemory()
        st.session_state.display_history = []
        st.rerun()
for role, text, candidates in st.session_state.display_history:
    with st.chat_message(role):
        st.markdown(text)
        if candidates:
            _render_candidates(candidates)
user_message = st.chat_input("What are you in the mood to watch?")
if user_message:
    st.session_state.display_history.append(("user", user_message, None))
    with st.chat_message("user"):
        st.markdown(user_message)
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            result = rag_pipeline.answer(
                user_message, st.session_state.memory, language=language
            )
        st.markdown(result["reply"])
        candidates = result["candidates"]
        if candidates:
            _render_candidates(candidates)
    st.session_state.display_history.append(("assistant", result["reply"], candidates))