mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
[theme]\n\
base="dark"\n\
primaryColor="#4833f6"\n\
backgroundColor="#000000"\n\
\n\
" > ~/.streamlit/config.toml
