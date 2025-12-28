"""
URL-Based Phishing Detection Module for PhishFusion-Net
========================================================

This module implements URL-based features for phishing detection including:
- Lexical analysis (URL length, special characters, entropy)
- Domain age and registration analysis
- SSL/HTTPS validation
- Suspicious pattern detection (homograph attacks, IP addresses)
- Brand impersonation detection
- Redirect chain analysis
"""

import re
import socket
import ssl
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import tldextract
import requests
from urllib.parse import urlparse
import math
from collections import Counter
import ipaddress
import warnings
import unicodedata

# Suppress SSL and urllib3 warnings for phishing site analysis
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class URLAnalyzer:
    """
    Comprehensive URL analysis for phishing detection
    Enhanced version with fixes for production use
    """
    
    def __init__(self, timeout=5, enable_whois=False, enable_dnssec=False):
        """
        Initialize URL Analyzer
        
        Args:
            timeout: Request timeout in seconds
            enable_whois: Enable WHOIS domain age checking (slower)
            enable_dnssec: Enable DNSSEC validation (requires dnspython)
        """
        self.timeout = timeout
        self.enable_whois = enable_whois
        self.enable_dnssec = enable_dnssec
        
        self.suspicious_tlds = [
            '.tk', '.ml', '.ga', '.cf', '.gq',  # Free domains
            '.xyz', '.top', '.work', '.click', '.link',  # Suspicious TLDs
            '.loan', '.racing', '.download', '.stream'
        ]
        
        # Homograph attack characters (lookalike characters)
        self.homograph_map = {
            'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',  # Cyrillic
            'х': 'x', 'у': 'y', 'і': 'i',
            'ı': 'i', 'ο': 'o', 'ν': 'v',  # Greek
        }
        
        # Common brand keywords for impersonation detection
        self.brand_keywords = [
            # Payment services
            'paypal', 'stripe', 'square', 'venmo', 'cashapp',
            # E-commerce
            'amazon', 'ebay', 'alibaba', 'etsy', 'shopify',
            # Tech giants
            'google', 'microsoft', 'apple', 'meta', 'facebook',
            'instagram', 'twitter', 'linkedin', 'youtube',
            # Streaming/Entertainment
            'netflix', 'spotify', 'hulu', 'disney', 'twitch',
            # Banking/Finance
            'bank', 'banking', 'wellsfargo', 'chase', 'citibank',
            'payoneer', 'wise', 'revolut',
            # Security/Account keywords
            'secure', 'security', 'login', 'signin', 'verify',
            'account', 'update', 'confirm', 'suspended', 'locked',
            'alert', 'warning', 'urgent', 'action', 'required',
            # Crypto
            'coinbase', 'binance', 'crypto', 'wallet', 'blockchain'
        ]
        
        # Legitimate domains that shouldn't trigger impersonation warnings (EXPANDED)
        self.whitelisted_domains = {
            'google': ['google.com', 'youtube.com', 'gmail.com', 'gstatic.com', 'googleapis.com', 'google.co.uk'],
            'microsoft': ['microsoft.com', 'live.com', 'outlook.com', 'office.com', 'msn.com', 'windows.com', 'xbox.com'],
            'apple': ['apple.com', 'icloud.com', 'me.com', 'apple.co.uk', 'cdn-apple.com'],
            'amazon': ['amazon.com', 'amazonaws.com', 'cloudfront.net', 'amazon.co.uk', 'amazon.de', 'amazon.fr'],
            'facebook': ['facebook.com', 'fb.com', 'fbcdn.net', 'facebook.net'],
            'instagram': ['instagram.com', 'cdninstagram.com'],
            'paypal': ['paypal.com', 'paypal-communication.com', 'paypalobjects.com'],
            'netflix': ['netflix.com', 'nflxext.com', 'nflximg.net', 'nflxvideo.net', 'nflxso.net'],
            'spotify': ['spotify.com', 'scdn.co', 'spotify.map.fastly.net'],
            'linkedin': ['linkedin.com', 'licdn.com'],
            'twitter': ['twitter.com', 't.co', 'twimg.com'],
            'ebay': ['ebay.com', 'ebayimg.com', 'ebaystatic.com'],
            'alibaba': ['alibaba.com', 'alicdn.com'],
            'binance': ['binance.com', 'bn.com', 'binance.cloud'],
            'coinbase': ['coinbase.com', 'cb-web.net'],
            'chase': ['chase.com', 'jpmorganchase.com'],
            'wellsfargo': ['wellsfargo.com', 'wf.com'],
            'citibank': ['citibank.com', 'citi.com'],
            'stripe': ['stripe.com', 'stripe.network'],
            'square': ['square.com', 'squareup.com', 'sqspcdn.com'],
            'venmo': ['venmo.com'],
            'cashapp': ['cash.app', 'cash.me'],
        }

    def analyze(self, url: str) -> Dict:
        """
        Perform comprehensive URL analysis
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary containing all URL features and risk scores
        """
        import time
        start_time = time.time()
        
        features = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'analysis_version': '2.0',
        }
        
        try:
            # Lexical features
            features.update(self._extract_lexical_features(url))
            
            # Domain features
            features.update(self._extract_domain_features(url))
            
            # SSL/HTTPS features
            features.update(self._check_ssl_certificate(url))
            
            # Suspicious pattern detection
            features.update(self._detect_suspicious_patterns(url))
            
            # Typosquatting detection
            if features.get('domain'):
                typosquat = self._detect_typosquatting(features['domain'])
                if typosquat:
                    features['typosquatting'] = typosquat
            
            # Redirect analysis
            features.update(self._analyze_redirects(url))
            
            # Domain age check (if WHOIS enabled)
            if self.enable_whois and features.get('registered_domain'):
                features.update(self._check_domain_age(features['registered_domain']))
            
            # Calculate overall risk score
            features['risk_score'] = self._calculate_risk_score(features)
            features['risk_level'] = self._categorize_risk(features['risk_score'])
            
            # Add confidence score
            features['confidence'] = self._calculate_confidence(features)
            
            # Add threat categories
            features['threat_categories'] = self._identify_threat_categories(features)
            
        except Exception as e:
            features['error'] = str(e)
            features['risk_score'] = 0.5
            features['risk_level'] = 'unknown'
            features['confidence'] = 0.0
            features['threat_categories'] = []
        
        features['analysis_time'] = time.time() - start_time
        
        return features

    def _extract_lexical_features(self, url: str) -> Dict:
        """Extract lexical features from URL"""
        features = {}
        
        try:
            parsed = urlparse(url)
            hostname = parsed.netloc
            path = parsed.path
            query = parsed.query
            
            # Basic features
            features['url_length'] = len(url)
            features['hostname_length'] = len(hostname)
            features['path_length'] = len(path)
            
            # Character counts
            features['dot_count'] = url.count('.')
            features['dash_count'] = url.count('-')
            features['digit_count'] = sum(c.isdigit() for c in url)
            features['at_count'] = url.count('@')
            
            # Ratios
            features['digit_ratio'] = features['digit_count'] / max(len(url), 1)
            
            # Entropy
            features['url_entropy'] = self._calculate_entropy(url)
            
            # Subdomain analysis
            ext = tldextract.extract(url)
            subdomain_parts = [part for part in ext.subdomain.split('.') if part]
            features['subdomain_count'] = len(subdomain_parts)
            
            # Length checks
            features['suspicious_length'] = features['url_length'] > 75
            features['very_long_url'] = features['url_length'] > 100
            
        except Exception as e:
            features['lexical_error'] = str(e)
        
        return features

    def _extract_domain_features(self, url: str) -> Dict:
        """Extract domain-related features"""
        features = {}
        
        try:
            ext = tldextract.extract(url)
            
            features['domain'] = ext.domain
            features['tld'] = ext.suffix
            features['registered_domain'] = ext.registered_domain
            features['domain_length'] = len(ext.domain)
            
            # TLD checks
            features['is_suspicious_tld'] = ('.' + ext.suffix) in self.suspicious_tlds
            features['is_ip_address'] = self._is_ip_address(ext.domain)
            features['is_punycode'] = ext.domain.startswith('xn--')
            
        except Exception as e:
            features['domain_error'] = str(e)
        
        return features

    def _check_ssl_certificate(self, url: str) -> Dict:
        """Check SSL certificate validity"""
        features = {
            'uses_https': False,
            'valid_ssl': False,
            'ssl_checked': False
        }
        
        try:
            parsed = urlparse(url)
            features['uses_https'] = parsed.scheme == 'https'
            
            if not features['uses_https']:
                return features
            
            hostname = parsed.hostname
            if not hostname:
                return features
            
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((hostname, 443), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    if cert:
                        features['valid_ssl'] = True
                        features['ssl_checked'] = True
                        
        except Exception as e:
            features['ssl_error'] = str(e)[:100]
        
        return features

    def _detect_suspicious_patterns(self, url: str) -> Dict:
        """Detect suspicious patterns in URL"""
        features = {}
        
        try:
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()
            path = parsed.path.lower()
            full_url = url.lower()
            
            # Pattern checks
            features['has_at_symbol'] = '@' in hostname
            features['has_double_slash'] = '//' in path
            
            clean_hostname = hostname.split(':')[0].strip('[]')
            features['uses_ip_address'] = self._is_ip_address(clean_hostname)
            
            features['has_homograph'] = self._detect_homograph(hostname)
            features['brand_impersonation'] = self._detect_brand_impersonation(full_url)
            
            # URL shorteners (EXPANDED LIST)
            shorteners = [
                'bit.ly', 'tinyurl.com', 'goo.gl', 't.co',
                'ow.ly', 'short.io', 'rebrand.ly', 'is.gd',
                'buff.ly', 'adf.ly', 'bl.ink', 'lnkd.in',
                'tiny.cc', 'cutt.ly', 'shorturl.at', 'v.gd'
            ]
            features['is_shortened'] = any(s in hostname for s in shorteners)
            
            # Suspicious keywords
            suspicious_keywords = ['verify', 'account', 'update', 'confirm', 'secure']
            features['suspicious_keyword_count'] = sum(
                1 for keyword in suspicious_keywords if keyword in full_url
            )
            
            # Excessive subdomains
            ext = tldextract.extract(url)
            subdomain_parts = [part for part in ext.subdomain.split('.') if part]
            features['excessive_subdomains'] = len(subdomain_parts) > 3
            
        except Exception as e:
            features['pattern_error'] = str(e)
        
        return features

    def _analyze_redirects(self, url: str) -> Dict:
        """Analyze redirect chains with GET fallback"""
        features = {
            'redirect_count': 0,
            'final_url': url,
            'has_redirects': False,
            'redirect_changes_domain': False
        }
        
        try:
            # Try GET with stream first (some servers don't support HEAD)
            response = requests.get(
                url,
                allow_redirects=True,
                timeout=self.timeout,
                verify=False,
                stream=True  # Don't download full content
            )
            response.close()  # Close stream immediately
            
            if response.history:
                features['has_redirects'] = True
                features['redirect_count'] = len(response.history)
                features['final_url'] = response.url
                
                # Check if redirect changes domain
                try:
                    original_domain = tldextract.extract(url).registered_domain
                    final_domain = tldextract.extract(response.url).registered_domain
                    if original_domain and final_domain:
                        features['redirect_changes_domain'] = (original_domain != final_domain)
                except:
                    pass
                
        except Exception as e:
            features['redirect_error'] = str(e)[:100]
        
        return features

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        if not text:
            return 0.0
        
        counts = Counter(text)
        length = len(text)
        
        entropy = 0.0
        for count in counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy

    def _is_ip_address(self, text: str) -> bool:
        """Check if text is an IP address"""
        try:
            ipaddress.ip_address(text)
            return True
        except ValueError:
            return False

    def _detect_homograph(self, text: str) -> bool:
        """Detect homograph attack"""
        for char in text:
            if char in self.homograph_map:
                return True
            if ord(char) > 127:
                try:
                    char_name = unicodedata.name(char, '')
                    suspicious_scripts = ['CYRILLIC', 'GREEK', 'ARABIC']
                    if any(script in char_name for script in suspicious_scripts):
                        return True
                except:
                    pass
        return False
    
    def _detect_typosquatting(self, domain: str) -> Optional[str]:
        """Detect typosquatting"""
        import difflib
        
        major_brands = [
            'google', 'paypal', 'amazon', 'microsoft', 'apple',
            'facebook', 'instagram', 'netflix', 'linkedin'
        ]
        
        domain_lower = domain.lower()
        
        for brand in major_brands:
            if domain_lower == brand:
                continue
            
            similarity = difflib.SequenceMatcher(None, domain_lower, brand).ratio()
            
            if 0.75 <= similarity < 1.0:
                return f"{brand} (typosquatting)"
        
        return None

    def _detect_brand_impersonation(self, url: str) -> Optional[str]:
        """Detect brand impersonation"""
        url_lower = url.lower()
        ext = tldextract.extract(url)
        registered_domain = ext.registered_domain.lower()
        
        for brand in self.brand_keywords:
            if brand not in url_lower:
                continue
            
            # Check whitelist
            if brand in self.whitelisted_domains:
                if registered_domain in self.whitelisted_domains[brand]:
                    continue
            
            # Brand in domain but not exact match
            if brand in ext.domain.lower() and ext.domain.lower() != brand:
                return brand
            
            # Brand in path
            parsed = urlparse(url_lower)
            if brand in parsed.path:
                return brand
        
        return None

    def _calculate_risk_score(self, features: Dict) -> float:
        """Calculate overall risk score (0-1) with AGGRESSIVE weighting for phishing detection"""
        positive_score = 0.0  # Risk indicators
        negative_score = 0.0  # Security indicators
        
        # CRITICAL POSITIVE RISK INDICATORS (Recalibrated for better detection)
        risk_checks = [
            ('uses_ip_address', 0.40),          # ↑ from 0.15 - IP address is extreme red flag
            ('is_ip_address', 0.40),            # Duplicate check for safety
            ('brand_impersonation', 0.25),      # ↑ from 0.12 - MOST CRITICAL signal
            ('has_homograph', 0.30),            # ↑ from 0.15 - Punycode attack
            ('typosquatting', 0.20),            # ↑ from 0.12 - Domain similarity attack
            ('is_suspicious_tld', 0.15),        # ↑ from 0.08 - .tk, .ml are red flags
            ('excessive_subdomains', 0.12),     # ↑ from 0.07 - a.b.c.d.e.com
            ('has_double_slash', 0.08),         # ↑ from 0.05
            ('redirect_changes_domain', 0.10),  # ↑ from 0.08
            ('is_shortened', 0.06),             # ↑ from 0.04
            ('has_at_symbol', 0.15),            # ↑ from 0.10
            ('suspicious_length', 0.08),        # ↑ from 0.05
        ]
        
        for feature, weight in risk_checks:
            if features.get(feature):
                positive_score += weight
        
        # Suspicious keywords bonus (NEW!)
        keyword_count = features.get('suspicious_keyword_count', 0)
        if keyword_count > 0:
            positive_score += min(keyword_count * 0.08, 0.25)  # Max +0.25 for keywords
        
        # NEGATIVE RISK INDICATORS (Security signals - reduce risk)
        if features.get('uses_https') and features.get('valid_ssl'):
            negative_score += 0.15  # ↑ from 0.10 - Valid HTTPS is strong positive
        elif not features.get('uses_https'):
            positive_score += 0.15  # ↑ from 0.08 - No HTTPS is very suspicious
        
        # Domain age bonus (if available)
        if features.get('domain_age_days', -1) > 365:
            negative_score += 0.08  # ↑ from 0.05 - Old domains more trustworthy
        elif features.get('domain_age_days', -1) >= 0 and features.get('domain_age_days') < 90:
            positive_score += 0.15  # ↑ from 0.10 - Very new domains suspicious
        
        # Calculate final score
        final_score = positive_score - negative_score
        return max(0.0, min(1.0, final_score))

    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level"""
        if score >= 0.7:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        elif score >= 0.3:
            return 'low'
        else:
            return 'safe'
    
    def _calculate_confidence(self, features: Dict) -> float:
        """Calculate confidence score based on analysis completeness"""
        checks_performed = 0
        checks_available = 0
        
        # Track which analyses completed successfully
        analysis_blocks = [
            'url_length',  # Lexical analysis
            'domain',  # Domain extraction
            'ssl_checked',  # SSL validation attempted
        ]
        
        for block in analysis_blocks:
            checks_available += 1
            if block in features:
                checks_performed += 1
        
        # Bonus for successful complex checks
        if features.get('ssl_checked'):
            checks_performed += 0.5
        
        if features.get('redirect_count') is not None:
            checks_performed += 0.5
        
        # Penalty for errors
        error_count = sum(1 for k in features if 'error' in k)
        error_penalty = min(error_count * 0.15, 0.5)  # Max 50% penalty
        
        # Calculate base confidence
        if checks_available > 0:
            base_confidence = checks_performed / (checks_available + 1)
        else:
            base_confidence = 0.5
        
        # Apply error penalty
        final_confidence = max(0.0, min(1.0, base_confidence - error_penalty))
        
        return round(final_confidence, 3)
    
    def _identify_threat_categories(self, features: Dict) -> List[str]:
        """Identify threat categories"""
        categories = []
        
        if features.get('brand_impersonation'):
            categories.append('brand_impersonation')
        if features.get('is_ip_address'):
            categories.append('ip_address_abuse')
        if features.get('has_homograph'):
            categories.append('homograph_attack')
        if features.get('typosquatting'):
            categories.append('typosquatting')
        
        return categories

    def _check_domain_age(self, domain: str) -> Dict:
        """
        Check domain age using WHOIS
        Requires: pip install python-whois
        """
        features = {
            'domain_age_days': -1,
            'domain_age_suspicious': False,
            'whois_checked': True
        }
        
        if not self.enable_whois:
            features['whois_checked'] = False
            return features
        
        try:
            import whois
            w = whois.whois(domain)
            
            if w and w.creation_date:
                # Handle list or single date
                creation_date = w.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                
                if creation_date:
                    # Make timezone-naive for compatibility
                    if hasattr(creation_date, 'tzinfo') and creation_date.tzinfo is not None:
                        creation_date = creation_date.replace(tzinfo=None)
                    
                    age_days = (datetime.now() - creation_date).days
                    features['domain_age_days'] = age_days
                    
                    # Domains less than 1 year old are somewhat suspicious
                    # Domains less than 90 days are very suspicious
                    features['domain_age_suspicious'] = age_days < 365
                    features['is_very_new_domain'] = age_days < 90
                    
        except ImportError:
            features['whois_error'] = 'python-whois not installed'
            features['whois_checked'] = False
        except Exception as e:
            features['whois_error'] = str(e)[:100]
        
        return features

    def get_summary(self, features: Dict) -> str:
        """Generate human-readable summary"""
        risk_score = features.get('risk_score', 0)
        risk_level = features.get('risk_level', 'unknown')
        
        summary = f"URL Risk: {risk_score:.2f} ({risk_level.upper()})\n"
        
        if features.get('brand_impersonation'):
            summary += f"⚠️ Brand impersonation: {features['brand_impersonation']}\n"
        if features.get('typosquatting'):
            summary += f"⚠️ {features['typosquatting']}\n"
        if features.get('is_suspicious_tld'):
            summary += f"⚠️ Suspicious TLD: {features.get('tld')}\n"
        
        return summary


def quick_url_check(url: str) -> Tuple[float, str]:
    """Quick URL check - returns (risk_score, risk_level)"""
    analyzer = URLAnalyzer(timeout=3)
    features = analyzer.analyze(url)
    return features['risk_score'], features['risk_level']
