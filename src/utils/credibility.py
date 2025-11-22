"""Source credibility scoring based on domain authority and other factors."""

import re
from typing import List, Dict, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class CredibilityScorer:
    """Score sources based on domain authority and other credibility factors."""
    
    # Trusted domains
    TRUSTED_DOMAINS = {
        # Academic institutions (global)
        '.edu', '.ac.uk', '.ac.in', '.edu.in', '.edu.au', '.ac.jp',
        
        # Government (global)
        '.gov', '.gov.uk', '.gov.au', '.gov.ca', '.gov.in', '.europa.eu',
        
        # International news organizations
        'bbc.com', 'bbc.co.uk', 'reuters.com', 'ap.org', 'npr.org',
        'theguardian.com', 'nytimes.com', 'washingtonpost.com', 'wsj.com',
        'ft.com', 'economist.com', 'bloomberg.com', 'cnbc.com',
        'cnn.com', 'aljazeera.com', 'france24.com', 'dw.com',
        
        # Indian news organizations
        'thehindu.com', 'indianexpress.com', 'timesofindia.com', 'indiatimes.com',
        'economictimes.com', 'financialexpress.com', 'livemint.com',
        'business-standard.com', 'moneycontrol.com', 'businessline.in',
        'businesstoday.in', 'businessinsider.in',
        
        # Academic & Research platforms
        'arxiv.org', 'scholar.google.com', 'researchgate.net', 'semanticscholar.org',
        'pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'nih.gov', 'nature.com',
        'sciencedirect.com', 'springer.com', 'wiley.com', 'ieee.org',
        'jstor.org', 'plos.org', 'sciencemag.org', 'cell.com',
        
        # Medical & Health organizations
        'who.int', 'cdc.gov', 'mayoclinic.org', 'nih.gov', 'webmd.com',
        
        # International organizations
        'un.org', 'worldbank.org', 'imf.org', 'wto.org', 'oecd.org',
        
        # Tech & Science publications
        'nature.com', 'scientificamerican.com', 'newscientist.com',
        'technologyreview.com', 'spectrum.ieee.org', 'arstechnica.com',
        'wired.com', 'techcrunch.com', 'theverge.com',
        
        # Wikipedia & educational resources
        'wikipedia.org', 'britannica.com', 'khanacademy.org',
        
        # Legal & policy
        'supremecourt.gov', 'congress.gov', 'loc.gov',
        
        # Statistics & data
        'census.gov', 'bls.gov', 'data.gov', 'worldbank.org',
        'statista.com', 'pewresearch.org', 'gallup.com'
    }
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r'\.(xyz|tk|ml|ga|cf|gq)$',  # Suspicious TLDs
        r'bit\.ly|tinyurl|t\.co',  # URL shorteners
        r'blogspot|wordpress\.com',  # Personal blogs (lower credibility)
    ]
    
    def score_url(self, url: str) -> Dict[str, Any]:
        """Score a URL's credibility.
        
        Returns:
            Dict with 'score' (0-100), 'factors', and 'level' (low/medium/high)
        """
        if not url:
            return {'score': 0, 'factors': ['No URL'], 'level': 'low'}
        
        score = 50  # Base score
        factors = []
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check for trusted domains
            is_trusted = False
            for trusted in self.TRUSTED_DOMAINS:
                if trusted in domain:
                    score += 30
                    factors.append(f'Trusted domain: {trusted}')
                    is_trusted = True
                    break
            
            # Check for suspicious patterns
            is_suspicious = False
            for pattern in self.SUSPICIOUS_PATTERNS:
                if re.search(pattern, domain):
                    score -= 20
                    factors.append(f'Suspicious pattern: {pattern}')
                    is_suspicious = True
                    break
            
            # HTTPS bonus
            if parsed.scheme == 'https':
                score += 5
                factors.append('HTTPS enabled')
            else:
                score -= 10
                factors.append('No HTTPS')
            
            # Domain age indicators (heuristic based on domain structure)
            if not is_trusted and not is_suspicious:
                # Longer domains might be less credible (often spam)
                if len(domain.split('.')) > 3:
                    score -= 5
                    factors.append('Complex domain structure')
            
            # Academic paths
            if '/papers/' in parsed.path or '/research/' in parsed.path or '/publications/' in parsed.path:
                score += 10
                factors.append('Academic/research path')
            
            # Normalize score to 0-100
            score = max(0, min(100, score))
            
            # Determine level
            if score >= 70:
                level = 'high'
            elif score >= 40:
                level = 'medium'
            else:
                level = 'low'
            
            return {
                'score': score,
                'factors': factors if factors else ['Standard domain'],
                'level': level,
                'domain': domain
            }
            
        except Exception as e:
            logger.warning(f"Error scoring URL {url}: {e}")
            return {'score': 30, 'factors': ['Scoring error'], 'level': 'low'}
    
    def score_search_results(self, results: List) -> List[Dict]:
        """Score a list of search results."""
        scored = []
        for result in results:
            if hasattr(result, 'url'):
                url = result.url
            elif isinstance(result, dict):
                url = result.get('url', '')
            else:
                url = str(result)
            
            credibility = self.score_url(url)
            scored.append({
                'result': result,
                'credibility': credibility
            })
        
        # Sort by credibility score (highest first)
        scored.sort(key=lambda x: x['credibility']['score'], reverse=True)
        return scored
    
    def filter_by_credibility(self, results: List, min_score: int = 40) -> List:
        """Filter results by minimum credibility score."""
        scored = self.score_search_results(results)
        filtered = [
            item['result'] for item in scored
            if item['credibility']['score'] >= min_score
        ]
        logger.info(f"Filtered {len(results)} -> {len(filtered)} results (min_score={min_score})")
        return filtered

