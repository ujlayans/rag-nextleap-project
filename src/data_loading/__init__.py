"""Phase 1: Data Loading — fetch and parse Groww scheme pages."""

from src.data_loading.fetcher import SCHEME_URLS, fetch_scheme_pages
from src.data_loading.parser import parse_html_to_markdown

__all__ = ["SCHEME_URLS", "fetch_scheme_pages", "parse_html_to_markdown"]
