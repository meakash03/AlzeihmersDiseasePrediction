import io

from flask import Flask, jsonify, request
from flask_cors import CORS
from predict import prediction

app = Flask(__name__)

app.config.from_object(__name__)

CORS(app, resources={r"/*":{'origins':"*"}})

def load_fasta(file):
    sequences = []
    current_sequence = ""
    for line in file:
        line = line.strip()
        if line.startswith('>'):
            if current_sequence:
                sequences.append(current_sequence)
            current_sequence = ""
        else:
            current_sequence += line
    if current_sequence:
        sequences.append(current_sequence)
    return sequences

@app.route('/process-file', methods=['POST','GET'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        sequence = load_fasta(io.StringIO(file.read().decode('utf-8')))
        output = prediction(sequence)  # Pass file content to your function
        return jsonify({'output': output})
    except Exception as e:
        print(f'Error processing file: {e}')
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)