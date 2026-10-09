"""NovaMind — Entry Point"""
import subprocess
import sys
import os

def main():
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Start the server
    from server import app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
