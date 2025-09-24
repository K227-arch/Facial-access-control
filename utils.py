# utils.py

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color string"""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)