"""
Phishing URL Detector — Streamlit App
Features engineered per professor's project specification:
  - URL length & depth
  - Presence of '@', '-', subdomains
  - HTTPS availability
  - Domain age (via WHOIS)
  - Number of redirections
  - Page title mismatch
  - Presence of login forms
  - Favicon source domain

Tools used: Python 3.x, Requests, BeautifulSoup4, whois, Scikit-learn, XGBoost, Streamlit
Dataset: PhiUSIIL Phishing URL Dataset (UCI / Kaggle)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import whois
from datetime import datetime

import pickle
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🎣",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_url_features(url: str, tld_prob_lookup: dict, tld_encoder) -> dict:
    """
    Extract URL-string features that require no page fetching.
    Covers: URL length & depth, '@' / '-' presence, subdomains, HTTPS, TLD stats.
    """
    parsed = urlparse(url)
    domain = parsed.netloc

    f = {}
    f['URLLength']      = len(url)
    f['DomainLength']   = len(domain)
    f['URLDepth']       = url.count('/')           # slashes in full URL (2 = https://<domain>)

    f['HasAtSymbol']     = 1 if '@' in url else 0
    f['HasDashInDomain'] = 1 if '-' in domain else 0
    f['NoOfSubDomain']   = max(0, domain.count('.') - 1)

    f['IsHTTPS'] = 1 if parsed.scheme == 'https' else 0

    tld = domain.split('.')[-1] if '.' in domain else ''
    f['TLD']             = tld
    f['TLDLength']       = len(tld)
    f['IsDomainIP']      = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0

    f['TLDLegitimateProb'] = tld_prob_lookup.get(tld.lower(), 0.0)
    try:
        f['TLD_encoded'] = int(tld_encoder.transform([tld])[0])
    except (ValueError, Exception):
        f['TLD_encoded'] = 0

    # Obfuscation: hex-encoded characters (%XX)
    obf = re.findall(r'%[0-9a-fA-F]{2}', url)
    f['HasObfuscation']     = 1 if obf else 0
    f['NoOfObfuscatedChar'] = len(obf)

    # URL query-string chars
    f['NoOfQMarkInURL']     = url.count('?')
    f['NoOfAmpersandInURL'] = url.count('&')
    f['NoOfEqualsInURL']    = url.count('=')

    return f


def is_spa(response, soup) -> bool:
    """
    Detect JavaScript-rendered Single Page Applications.
    SPAs return a near-empty HTML shell; BeautifulSoup/requests cannot see JS content.
    Heuristics: very few links + scripts present + almost no readable body text, OR
    a known SPA root div, OR a single-line HTML response.
    """
    if response is None:
        return False
    all_links  = soup.find_all('a', href=True)
    all_scripts = soup.find_all('script')
    body       = soup.find('body')
    body_text  = body.get_text(strip=True) if body else ''
    lines      = response.text.split('\n')

    spa_roots = soup.find_all(
        ['div', 'section'],
        id=lambda x: x in ['root', '__next', 'app', 'main', 'app-root', 'gatsby-focus-wrapper']
    )
    return (
        (len(all_links) < 5 and len(all_scripts) >= 1 and len(body_text) < 500)
        or len(spa_roots) > 0
        or (len(lines) <= 3 and len(all_scripts) >= 1)
    )


def extract_html_features(url: str) -> dict:
    """
    Fetch the page and extract HTML-based features:
      - Page title mismatch (DomainTitleMatchScore)
      - Login form indicators (password field, submit button, hidden fields)
      - Favicon presence
      - iFrame, popup, external form action
      - Keyword flags (bank, pay, crypto)
      - Number of redirections
    Returns a dict plus '_is_spa' (internal flag, not passed to model).
    """
    domain  = urlparse(url).netloc
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    try:
        response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        redirects = len(response.history)
        html      = response.text
        soup      = BeautifulSoup(html, 'html.parser')
    except Exception:
        return {
            '_is_spa': False, 'NoOfURLRedirect': 0,
            'DomainTitleMatchScore': 0, 'HasTitle': 0,
            'HasPasswordField': 0, 'HasSubmitButton': 0,
            'HasHiddenFields': 0, 'HasExternalFormSubmit': 0,
            'HasFavicon': 0, 'NoOfiFrame': 0,
            'Bank': 0, 'Pay': 0, 'Crypto': 0,
        }

    f = {}
    f['_is_spa']         = is_spa(response, soup)
    f['NoOfURLRedirect'] = redirects

    # Page title mismatch
    title_tag = soup.find('title')
    title     = title_tag.text.strip() if title_tag else ''
    f['HasTitle'] = 1 if title else 0

    domain_core = domain.replace('www.', '').split('.')[0]
    f['DomainTitleMatchScore'] = 100 if domain_core.lower() in title.lower() else 0

    # Favicon (professor spec: favicon source domain)
    favicon_tag = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
    if favicon_tag:
        favicon_href = favicon_tag.get('href', '')
        # If favicon is on a completely different domain, treat as external (suspicious)
        favicon_external = (
            favicon_href.startswith('http') and domain not in favicon_href
        )
        f['HasFavicon'] = 0 if favicon_external else 1
    else:
        f['HasFavicon'] = 0

    # Login form features (professor spec: presence of login forms)
    f['HasPasswordField']    = 1 if soup.find('input', type='password') else 0
    f['HasSubmitButton']     = 1 if (
        soup.find('input', type='submit') or soup.find('button', type='submit')
    ) else 0
    f['HasHiddenFields']     = 1 if soup.find('input', type='hidden') else 0
    forms = soup.find_all('form')
    f['HasExternalFormSubmit'] = 1 if any(
        fm.get('action', '').startswith('http') and domain not in fm.get('action', '')
        for fm in forms
    ) else 0

    # iFrames and popups
    f['NoOfiFrame'] = len(soup.find_all('iframe'))

    # Keyword flags in page content
    hl = html.lower()
    f['Bank']   = 1 if 'bank'   in hl else 0
    f['Pay']    = 1 if 'pay'    in hl else 0
    f['Crypto'] = 1 if 'crypto' in hl else 0

    return f


def extract_whois_age(url: str) -> int:
    """
    Domain age (in days) via WHOIS — professor's spec: 'Domain age (via WHOIS)'.
    Returns -1 if unavailable.
    """
    try:
        domain = urlparse(url).netloc.replace('www.', '')
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            if creation.tzinfo is not None:
                creation = creation.replace(tzinfo=None)
            return (datetime.now() - creation).days
    except Exception:
        pass
    return -1


def extract_all_features(url: str, tld_prob_lookup: dict, tld_encoder) -> dict:
    """Combine URL, HTML, and WHOIS features into one dict."""
    url_feats  = extract_url_features(url, tld_prob_lookup, tld_encoder)
    html_feats = extract_html_features(url)
    domain_age = extract_whois_age(url)

    return {**url_feats, **html_feats, 'DomainAge': domain_age}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    import os
    base       = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, '..', 'models')

    with open(os.path.join(models_dir, 'best_model_v2.pkl'),        'rb') as f: model       = pickle.load(f)
    with open(os.path.join(models_dir, 'scaler_v2.pkl'),            'rb') as f: scaler      = pickle.load(f)
    with open(os.path.join(models_dir, 'tld_encoder_v2.pkl'),       'rb') as f: tld_encoder = pickle.load(f)
    with open(os.path.join(models_dir, 'feature_columns_v2.json'),  'r')  as f: fc          = json.load(f)
    with open(os.path.join(models_dir, 'tld_prob_lookup.json'),     'r')  as f: tld_lookup  = json.load(f)

    return model, scaler, tld_encoder, fc['feature_columns'], tld_lookup

model, scaler, tld_encoder, feature_columns, tld_prob_lookup = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎣 Phishing URL Detector")
st.write("Enter any URL to check whether it's likely **legitimate** or a **phishing** attempt.")
st.caption(
    "Features used: URL depth, HTTPS, subdomains, '@'/'-' symbols, "
    "title mismatch, login forms, favicon, redirects, TLD reputation, obfuscation."
)

url_input = st.text_input("URL to scan", placeholder="https://example.com")

if st.button("🔍 Scan URL", type="primary"):
    if not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        # Normalize: add scheme if missing
        url = url_input.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        with st.spinner("Fetching page and extracting features… (may take 10–20 s)"):
            try:
                raw = extract_all_features(url, tld_prob_lookup, tld_encoder)

                # Build feature vector in the exact column order the model expects
                feat_dict = {col: raw.get(col, 0) for col in feature_columns}
                X_input   = pd.DataFrame([feat_dict])[feature_columns]
                X_scaled  = scaler.transform(X_input)

                prediction = model.predict(X_scaled)[0]
                proba      = model.predict_proba(X_scaled)[0]

                spa_detected = raw.get('_is_spa', False)

                st.divider()

                if prediction == 1:
                    conf = proba[1] * 100
                    st.success(f"✅ Likely **LEGITIMATE** ({conf:.1f}% confidence)")
                else:
                    conf = proba[0] * 100
                    st.error(f"⚠️ Likely **PHISHING** ({conf:.1f}% confidence)")

                if spa_detected:
                    st.warning(
                        "⚠️ **JavaScript-rendered site detected.** This site loads its content "
                        "via JavaScript (React / Next.js / Vue). Our static scraper reads only "
                        "the HTML shell, so HTML-based features (title, favicon, forms) may be "
                        "incomplete. Treat this prediction with extra caution."
                    )

                # Key features summary
                with st.expander("📊 Feature summary"):
                    summary = {
                        "URL length":              raw.get('URLLength'),
                        "URL depth (slash count)": raw.get('URLDepth'),
                        "HTTPS":                   bool(raw.get('IsHTTPS')),
                        "Subdomains":              raw.get('NoOfSubDomain'),
                        "Has '@' in URL":          bool(raw.get('HasAtSymbol')),
                        "Has '-' in domain":       bool(raw.get('HasDashInDomain')),
                        "Domain age (days)":       raw.get('DomainAge'),
                        "Redirections":            raw.get('NoOfURLRedirect'),
                        "Title present":           bool(raw.get('HasTitle')),
                        "Title matches domain":    bool(raw.get('DomainTitleMatchScore')),
                        "Favicon present":         bool(raw.get('HasFavicon')),
                        "Password field":          bool(raw.get('HasPasswordField')),
                        "Submit button":           bool(raw.get('HasSubmitButton')),
                        "Hidden fields":           bool(raw.get('HasHiddenFields')),
                        "External form action":    bool(raw.get('HasExternalFormSubmit')),
                        "TLD reputation":          round(raw.get('TLDLegitimateProb', 0), 4),
                        "IP as domain":            bool(raw.get('IsDomainIP')),
                        "Obfuscated chars":        raw.get('NoOfObfuscatedChar'),
                        "iFrames":                 raw.get('NoOfiFrame'),
                        "Bank / Pay / Crypto":     f"{bool(raw.get('Bank'))} / {bool(raw.get('Pay'))} / {bool(raw.get('Crypto'))}",
                    }
                    for k, v in summary.items():
                        st.text(f"{k:<35} {v}")

                with st.expander("🔬 All raw extracted features"):
                    display = {k: v for k, v in raw.items() if not k.startswith('_')}
                    st.json(display)

            except Exception as e:
                st.error(f"Could not analyze this URL: {e}")
                import traceback
                st.code(traceback.format_exc())

st.divider()
st.caption("Built for NIT internship — Project 02: AI-Powered Phishing Website Detection")