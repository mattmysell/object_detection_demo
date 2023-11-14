#!/usr/bin/env python3
"""
Code running the project.
"""

# Standard Libraries

# Installed Libraries

# Local Files
from app.endpoints import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
