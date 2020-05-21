import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uncertify
from uncertify.common import PROJECT_ROOT_PATH
NOTEBOOKS_TEST_ASSETS_PATH = PROJECT_ROOT_PATH / 'notebooks' / 'assets'


"""
Now we can do 
    from context import uncertify
inside our notebooks inside this folder and subsequently
    from uncertify import <SOMETHING>
"""
