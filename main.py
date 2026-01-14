import streamlit as st
from rag import process_urls, generate_answers,format_context

st.title("Real-Estate-Assistant")

# --- SIDEBAR: URL INDEXING ---
st.sidebar.header("Index the URLs")
url1 = st.sidebar.text_input('URL1')
url2 = st.sidebar.text_input('URL2')
url3 = st.sidebar.text_input('URL3')

if st.sidebar.button("Process URLs"):
    urls = [url for url in (url1, url2, url3) if url != '']
    if not urls:
        st.sidebar.error("Provide at least one URL")
    else:
        with st.sidebar.status("Indexing...", expanded=True) as status:
            for step in process_urls(urls):
                st.write(step)
            status.update(label="Indexing Complete!", state="complete")

# --- CHAT HISTORY INITIALIZATION ---
# This keeps the chat iterative
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT ---
if query := st.chat_input("Enter your question"):
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    # 2. Add user message to session state
    st.session_state.messages.append({"role": "user", "content": query})

    # 3. Generate response
    try:
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            full_response = ""
            
            # Iterate through your generator
            for step in generate_answers(query):
                if step.startswith(">>"):
                    status_placeholder.text(step) # Show progress logs
                else:
                    full_response = step # This is the final string answer
            
            status_placeholder.empty() # Remove logs once we have the answer
            st.markdown(full_response)
            
        # 4. Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        st.error(f"Error: {e}")