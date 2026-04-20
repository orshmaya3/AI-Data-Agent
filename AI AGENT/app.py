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
<div style="min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px 20px;font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;">
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:52px 48px;max-width:480px;width:100%;">
        <svg width="64" height="64" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:32px;">
            <circle cx="36" cy="36" r="26" stroke="rgba(120,160,255,0.3)" stroke-width="1.5"/>
            <ellipse cx="36" cy="36" rx="26" ry="11" stroke="rgba(120,160,255,0.15)" stroke-width="1.2"/>
            <circle cx="36" cy="10" r="5" fill="#78a0ff" style="filter:drop-shadow(0 0 8px rgba(120,160,255,0.7))"/>
            <circle cx="36" cy="36" r="6" fill="rgba(120,160,255,0.2)" stroke="rgba(120,160,255,0.4)" stroke-width="1"/>
        </svg>
        <div style="display:inline-block;font-size:11px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:rgba(120,160,255,0.75);background:rgba(120,160,255,0.08);border:1px solid rgba(120,160,255,0.2);border-radius:20px;padding:5px 14px;margin-bottom:24px;">Hi, I'm Taeir</div>
        <div style="font-size:38px;font-weight:700;color:rgba(240,240,250,0.92);line-height:1.15;margin-bottom:16px;letter-spacing:-0.5px;">This project has<br>a new home.</div>
        <div style="font-size:16px;color:rgba(240,240,250,0.5);margin-bottom:40px;line-height:1.65;">If you're a recruiter exploring my work —<br>you're in the right place, just one click away.</div>
        <a href="https://ortaeir.com" target="_blank" style="display:inline-block;background:#78a0ff;color:#000;font-size:15px;font-weight:700;padding:14px 40px;border-radius:8px;text-decoration:none;letter-spacing:0.3px;" onmouseover="this.style.opacity='0.82'" onmouseout="this.style.opacity='1'">Visit ortaeir.com →</a>
        <div style="margin-top:20px;font-size:13px;color:rgba(240,240,250,0.22);letter-spacing:0.5px;">ortaeir.com</div>
    </div>
</div>
""", unsafe_allow_html=True)
