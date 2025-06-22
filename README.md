# AI Detector Test Model

This repository contains a notebook for testing a fine-tuned DistilBERT model that detects AI-generated text.

## Prerequisites

- Jupyter Notebook or JupyterLab

## Installation

1. Clone or download this repository to your local machine.

2. Navigate to the project directory:

   ```bash
   cd CS_162_Project
   ```

3. Install the required dependencies by running the first cell in the notebook, or manually install them:

   ```bash
   pip install datasets
   pip install -U accelerate
   pip install -U transformers
   pip install -U scikit-learn
   ```

   If you have a CUDA device

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   If no CUDA device is available

   ```bash
   pip install torch torchvision torchaudio
   ```

## Data Structure

Ensure your project directory has the following structure:

```
CS_162_Project/
├── test_model.ipynb
├── distilbert_ai_detector/          # Pre-trained model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
└── test_data/                        # Test data directory
    ├── test files...
```

## Test Data Format

The test data files should be in JSONL format with the following structure:

```json
{
  "human_text": "Human written text here",
  "machine_text": "AI generated text here"
}
```

## Running the Notebook

1. Open Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook
   ```

2. Navigate to and open `test_model.ipynb`

3. Run the cells in order:
   - **Cell 1**: Imports necessary libraries
   - **Cell 2**: Contains the main function that:
     - Loads the pre-trained DistilBERT model
     - Processes test data from the `test_data` directory
     - Tokenizes the text data
     - Evaluates the model performance
   - **Cell 3**: Executes the main function

## Model Evaluation

The notebook will output evaluation metrics including:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate
- **Recall**: Sensitivity of the model

## Customization

To test with different data:

1. Replace the files in the `test_data` directory with your own JSONL files
2. Update the `test_files` list in the main function to match your file names:
   ```python
   test_files = ['your_file1.jsonl', 'your_file2.jsonl', ...]
   ```

## Troubleshooting

- **Model not found**: Ensure the `distilbert_ai_detector` directory exists and contains the model files
- **Data not found**: Verify the `test_data` directory exists and contains the specified JSONL files
- **Memory issues**: Reduce batch size or use a smaller dataset if running into memory constraints
- **Import errors**: Make sure all required packages are installed correctly

## Expected Output

After running the notebook, you should see output similar to:

```
Evaluation results: {'eval_loss': 0.123, 'eval_accuracy': 0.95, 'eval_f1': 0.94, 'eval_precision': 0.93, 'eval_recall': 0.96}
```
