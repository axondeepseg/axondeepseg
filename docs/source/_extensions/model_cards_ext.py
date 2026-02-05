'''
This lightweight extension simply reads the model_cards.yaml file and creates a 
RST block to insert directly inside the documentation. Thus, to modify the model
list, we can simply edit the model_cards.yaml file and the doc will be auto-updated 
accordingly.
'''

from sphinx.util import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_yaml_simple(file_path):
    """
    Simple YAML parser that extracts top-level keys and model-info values.
    This avoids the dependency on PyYAML which may not be available to the
    RTD venv.
    """
    model_dict = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    current_model = None
    current_indent = -1
    
    for line in lines:
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith('#'):
            continue
        
        # Calculate indentation
        indent = len(line) - len(line.lstrip())
        
        # Check if this is a top-level model key (no indentation)
        if indent == 0 and ':' in line and not line.strip().startswith('-'):
            current_model = line.split(':')[0].strip()
            model_dict[current_model] = {'model-info': ''}
            current_indent = 0
        
        # Check if this is a model-info line
        elif current_model and 'model-info:' in line:
            # Extract the value after 'model-info:'
            value = line.split('model-info:', 1)[1].strip()
            model_dict[current_model]['model-info'] = value
    
    return model_dict

def get_model_cards(model_list_path=None):
    """Load the model list from a YAML file."""
    if model_list_path is None:
        # Get the path relative to this extension file
        model_list_path = Path(__file__).parent.parent.parent.parent / 'AxonDeepSeg' / 'model_cards.yaml'
    
    return parse_yaml_simple(model_list_path)

def generate_model_cards_rst(app, config):
    """Generate RST content from model_cards.yaml"""
    model_cards = get_model_cards()
    rst_content = ".. _model_cards:\n\n"
    rst_content += "The following models are available for download:\n\n"
    
    # Create RST table
    rst_content += ".. list-table::\n"
    rst_content += "   :header-rows: 1\n"
    rst_content += "   :widths: 20 80\n\n"
    rst_content += "   * - Model\n"
    rst_content += "     - Description\n"
    
    for model_key, model_info in model_cards.items():
        model_details = model_info.get('model-info', '')
        rst_content += f"   * - **{model_key}**\n"
        rst_content += f"     - {model_details}\n"
    
    # Write to a temporary RST file
    output_path = Path(app.confdir) / '_generated_models.rst'
    with open(output_path, 'w') as f:
        f.write(rst_content)
    
    logger.info(f"Generated models documentation at {output_path}")

def setup(app):
    app.connect('config-inited', generate_model_cards_rst)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }