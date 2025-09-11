import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# Load .env variables
load_dotenv()

# Model provider mapping
provider_map = {
    "OpenAI": "openai",
    "Gemini (Google)": "google_genai",
    "Anthropic (Claude)": "anthropic",
    "Mistral": "mistralai",
  
}

# Default models for each provider
default_models = {
    "OpenAI": "gpt-4o-mini",
    "Gemini (Google)": "gemini-2.0-flash",
    "Anthropic (Claude)": "claude-3-haiku-20240307",
    "Mistral": "mistral-large-latest",
}

# Streamlit UI
st.set_page_config(page_title="Smart News Summarizer", page_icon="📰")
st.title("📰 Smart News Summarizer")
st.markdown("---")
st.markdown(f"📅 **Date:** {datetime.now().strftime('%B %d, %Y')}")
st.markdown("🔍 Enter any topic to get a 5-bullet point summary!")

# Sidebar: Select provider
provider_label = st.sidebar.selectbox("🤖 Choose Model Provider", list(provider_map.keys()))
provider = provider_map[provider_label]

# API Key input
api_key = st.sidebar.text_input(f"🔑 Enter your {provider_label} API Key", type="password")

# Model name input
model_name = st.sidebar.text_input("🧠 Model Name", value=default_models[provider_label])

# Prompt template
system_template = """
You are an expert breaking news summarizer.

Given multiple latest web articles about a topic, extract and summarize the most **recent updates** into **5 clear bullet points**.

Each bullet point must be:
- Very recent (last few hours or days)
- Concise (maximum 2 lines)
- Focused on facts, dates, events, or announcements
- Professional, neutral, and informative

Avoid old information, speculation, opinions, and unnecessary storytelling.
Your goal is to quickly update the user with the **latest major developments**.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{search_results}")
])

# Digest generation function
def generate_digest(user_topic, model):
    search_tool = TavilySearchResults(max_results=5)
    results = search_tool.invoke(user_topic)

    if not results:
        return "❌ Sorry, no relevant information found for this topic."

    contents = "\n\n".join([item["content"] for item in results])
    prompt = prompt_template.invoke({"search_results": contents})
    return model.invoke(prompt).content

# Main input
user_input = st.text_input("Enter a topic (e.g., AI News, Cricket World Cup, Tesla Stock, etc.):")
# Button to trigger summary
if st.button("Fetch & Summarize 🚀"):
    if not api_key or not user_input.strip():
        st.warning("⚠️ Please enter both API key and a topic.")
    else:
        # Set environment variable based on provider
        key_env_map = {
            "openai": "OPENAI_API_KEY",
            "google_genai": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "mistralai": "MISTRAL_API_KEY",
        }
        os.environ[key_env_map[provider]] = api_key

        # Init model
        with st.spinner("🧠 Initializing model..."):
            model = init_chat_model(model_name, model_provider=provider)

        # Generate summary
        with st.spinner(f"🔍 Summarizing news about **{user_input}**..."):
            summary = generate_digest(user_input, model)
            st.success(f"✅ Summary for **{user_input}**:")
            st.markdown("### 🧠 Top 5 Bullet Points:")
            st.markdown(summary)

else:
    st.info("Enter a topic and click the button to generate your news summary.")
