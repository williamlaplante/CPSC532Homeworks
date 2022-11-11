import json
import subprocess
from os.path import exists

def load_program(daphne_dir, daphne_file, json_file, mode, compile=True):
    '''
    Either load a pre-compiled json file or compile daphne and save json
    '''
    if (not compile) and exists(json_file):
        with open(json_file) as f:
            ast_or_graph = json.load(f)
    else:
        ast_or_graph = get_json_from_daphne([mode, '-i', daphne_file], daphne_dir)
        with open(json_file, 'w') as f:
            json.dump(ast_or_graph, f, indent=4, ensure_ascii=False)
    return ast_or_graph


def get_json_from_daphne(args, daphne_dir):
    '''
    Run the daphne compiler and return the output in json format
    '''
    proc = subprocess.run(['lein', 'run', '-f', 'json']+args, capture_output=True, cwd=daphne_dir)
    if proc.returncode != 0:
        raise Exception(proc.stdout.decode()+proc.stderr.decode())
    return json.loads(proc.stdout) # Load the output into json