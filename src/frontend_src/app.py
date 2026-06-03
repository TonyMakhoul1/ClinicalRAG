import json

import httpx
import streamlit as st

from config.frontend_settings import FrontendSettings

settings = FrontendSettings()

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def show_login() -> None:
    st.title("📄 Document Q&A")
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
            return
        try:
            resp = httpx.post(
                settings.AUTH_URL,
                json={"username": username, "password": password},
                timeout=10.0,
            )
            if resp.status_code == 200:
                st.session_state.token = resp.json()["access_token"]
                st.session_state.messages = []
                st.rerun()
            elif resp.status_code == 401:
                st.error("Invalid username or password.")
            else:
                st.error(f"Login failed: {resp.status_code}")
        except Exception as e:
            st.error(f"Could not reach the backend: {e}")


if "token" not in st.session_state:
    st.session_state.token = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show login screen if not authenticated
if not st.session_state.token:
    show_login()
    st.stop()


st.title("📄 Document Q&A")
st.caption("Ask questions about your uploaded documents.")

# Logout button in the sidebar
with st.sidebar:
    st.write("Logged in")
    if st.button("Logout", use_container_width=True):
        st.session_state.token = None
        st.session_state.messages = []
        st.rerun()


def confidence_badge(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.80:
        color = "green"
    elif score >= 0.50:
        color = "orange"
    else:
        color = "red"
    return f'<span style="color:{color};font-weight:600">Confidence: {pct}%</span>'


# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("confidence") is not None:
                st.markdown(confidence_badge(
                    msg["confidence"]), unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**{src['source']}** — page {src['page']}")
                        st.caption(src["content"][:300])

# New user input
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        conf_placeholder = st.empty()
        sources = []
        confidence = None
        answer_tokens = []

        try:
            with httpx.stream(
                "POST",
                settings.CHAT_STREAM_URL,
                json={"query": prompt},
                headers=auth_headers(),
                timeout=60.0,
            ) as resp:
                # Token expired or invalid, log out and send user back to login
                if resp.status_code == 401:
                    st.session_state.token = None
                    st.error("Session expired. Please log in again.")
                    st.rerun()

                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event["type"] == "thinking":
                        placeholder.markdown(f"*{event['data']}*")
                    elif event["type"] == "confidence":
                        confidence = event["data"]
                        conf_placeholder.markdown(confidence_badge(
                            confidence), unsafe_allow_html=True)
                    elif event["type"] == "sources":
                        sources = event["data"]
                    elif event["type"] == "token":
                        answer_tokens.append(event["data"])
                        placeholder.markdown("".join(answer_tokens) + "▌")
                    elif event["type"] in ("error", "retract"):
                        answer_tokens = [event["data"]]
                        sources = []
                        confidence = None

            answer = "".join(answer_tokens)
            placeholder.markdown(answer)

        except httpx.HTTPStatusError as e:
            answer = f"Backend error: {e.response.status_code} — {e.response.text}"
            placeholder.markdown(answer)
        except Exception as e:
            answer = f"Could not reach the backend: {e}"
            placeholder.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"**{src['source']}** — page {src['page']}")
                    st.caption(src["content"][:300])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer,
            "sources": sources, "confidence": confidence}
    )
