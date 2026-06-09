import os
import click

# start with the click commands


@click.group()
def main():
    pass


@main.command()
def version():
    git_revision_short_hash = 'None'
    ssc4frames_version = 'None'
    try:
        import subprocess
        git_revision_short_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except:
        pass
    from importlib.metadata import version
    ssc4frames_version = f'ssc4frames.v{version("ssc4frames")}'
    print(f"Current git commit: {git_revision_short_hash}")
    print(f"SSC4frames version: {ssc4frames_version}")


# Import subcommand modules to register them with the main group
# This import must come AFTER main is defined to avoid circular imports
from ssc4frames.cli import data, clustering, experiment, experiments
