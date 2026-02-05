from sphinx.util import logging
import yaml
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def get_model_cards(model_list_path=None):
    """Load the model list from a YAML file."""
    if model_list_path is None:
        # Get the path relative to this extension file
        model_list_path = Path(__file__).parent.parent.parent.parent / 'AxonDeepSeg' / 'model_cards.yaml'
    
    with open(model_list_path, 'r') as f:
        model_dict = yaml.safe_load(f)
    return model_dict

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