import click
import subprocess

@click.command()
@click.option('--file', default='src/main.py', help='Path to the Streamlit application file.')
def run(file):
    """Run a Streamlit application."""
    subprocess.run(['streamlit', 'run', file])

if __name__ == '__main__':
    run()
