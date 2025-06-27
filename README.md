# Abstractive Text Summarization with Transformers and Gradio UI

## Project Description

This project develops an end-to-end abstractive text summarization system leveraging a Transformer-based neural network. The model is fine-tuned on a large-scale dataset (specifically the CNN/DailyMail news dataset) to generate concise, human-like summaries by rephrasing and condensing original text, rather than merely extracting sentences. A core feature of this project is the integration of a user-friendly web interface built with Gradio, which allows users to input articles and customize summarization parameters (such as minimum/maximum summary length and beam search strategy) in real-time. This project demonstrates proficiency in advanced Natural Language Processing (NLP) techniques, deep learning model fine-tuning, performance evaluation using ROUGE scores, and practical model deployment using modern MLOps tools.

## Features

  * **Abstractive Summarization:** Generates novel sentences and phrases to create summaries, going beyond simple extraction.
  * **Transformer Model:** Utilizes a powerful Transformer architecture (e.g., T5-small) for state-of-the-art sequence-to-sequence capabilities.
  * **Large Dataset Training:** Fine-tuned on the comprehensive CNN/DailyMail dataset for robust performance.
  * **Interactive Web UI:** Built with Gradio, providing an easy-to-use interface for users to submit text and receive summaries.
  * **Customizable Parameters:** Users can adjust `min_length`, `max_length`, and `num_beams` for tailored summary generation.
  * **GPU Acceleration:** Leverages PyTorch and Hugging Face `accelerate` for efficient training and inference on CUDA-enabled GPUs.

## Setup and Installation

To run this project, you'll need Python 3.8+ and the following libraries. It's highly recommended to use a virtual environment.


# Important: Handle fsspec dependency for Google Colab compatibility
# This helps resolve potential conflicts with pre-installed Colab packages like gcsfs.
pip install fsspec==2025.3.2
pip install -U huggingface_hub
```

## Usage

This project includes both the model training pipeline and a Gradio-based UI for inference.

1.  **Run the script:**

    ```bash
    python your_project_file_name.py # Replace with the actual name of your .py file
    ```

2.  **Training Process:**

      * The script will first download and preprocess the CNN/DailyMail dataset (this might take some time depending on your internet speed).
      * It will then begin the model training phase. **Note:** Training on the full dataset, especially with a base Transformer model, requires significant computational resources (preferably a GPU with at least 16GB VRAM) and can take several hours to days.
      * Training progress, including ROUGE scores on the validation set, will be logged in your console.
      * The best model checkpoint will be saved in the `./summarizer_results` directory.

3.  **Launch Gradio UI:**

      * After training (or if training fails, it will fall back to using the base pre-trained model), the Gradio UI will automatically launch.
      * A local URL (e.g., `http://127.0.0.1:7860`) will be printed in your console. If `GRADIO_SHARE` is set to `True`, a public shareable link (valid for 72 hours) will also be provided.
      * Open this URL in your web browser.

4.  **Interact with the UI:**

      * Paste your desired article into the "Input Article" text box.
      * Adjust the "Min Summary Length," "Max Summary Length," and "Number of Beams" sliders as desired.
      * The generated summary will appear in the "Generated Summary" box.

## Project Structure (Conceptual)

```
.
├── your_project_file_name.py  # Main script containing all code
├── summarizer_results/        # Directory for saved model checkpoints
│   └── checkpoint-XXXX/
│       ├── pytorch_model.bin
│       └── ...
└── summarizer_logs/           # Directory for training logs (TensorBoard compatible)
    └── run-YYYY-MM-DD_HH-MM-SS/
        └── ...
```

## Technologies Used

  * **Python**
  * **PyTorch**
  * **Hugging Face Transformers:** For pre-trained models and easy fine-tuning.
  * **Hugging Face Datasets:** For efficient data loading and preprocessing.
  * **Hugging Face Evaluate:** For robust metric computation (ROUGE).
  * **Gradio:** For rapid web interface development.
  * **NumPy**

