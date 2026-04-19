import streamlit as st

st.set_page_config(
    page_title="AI Data Department",
    page_icon="📊",
    layout="centered",
)

st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
    .stApp    { background: #000; }
    .block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
">
    <!-- Orbital logo -->
    <svg width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:28px;">
        <circle cx="36" cy="36" r="26" stroke="rgba(120,160,255,0.3)" stroke-width="1.5"/>
        <ellipse cx="36" cy="36" rx="26" ry="11" stroke="rgba(120,160,255,0.15)" stroke-width="1.2"/>
        <circle cx="36" cy="10" r="5" fill="#78a0ff" style="filter:drop-shadow(0 0 8px rgba(120,160,255,0.7))"/>
        <circle cx="36" cy="36" r="6" fill="rgba(120,160,255,0.2)" stroke="rgba(120,160,255,0.4)" stroke-width="1"/>
    </svg>

    <div style="font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                color:rgba(240,240,250,0.35);margin-bottom:20px;">
        Orbital AI · Data Agent
    </div>

    <div style="font-size:36px;font-weight:700;color:rgba(240,240,250,0.92);
                line-height:1.2;margin-bottom:14px;letter-spacing:-0.5px;">
        We moved to our own domain!
    </div>

    <div style="font-size:16px;color:rgba(240,240,250,0.45);margin-bottom:40px;max-width:420px;line-height:1.6;">
        This page is no longer active. The full platform is now live at our dedicated domain.
    </div>

    <a href="https://ortaeir.com" target="_blank" style="
        display: inline-block;
        background: #78a0ff;
        color: #000;
        font-size: 15px;
        font-weight: 700;
        padding: 14px 36px;
        border-radius: 6px;
        text-decoration: none;
        letter-spacing: 0.2px;
        transition: opacity 0.15s;
    " onmouseover="this.style.opacity='0.85'" onmouseout="this.style.opacity='1'">
        Visit ortaeir.com →
    </a>

    <div style="margin-top:20px;font-size:13px;color:rgba(240,240,250,0.25);">
        ortaeir.com
    </div>
</div>
""", unsafe_allow_html=True)
