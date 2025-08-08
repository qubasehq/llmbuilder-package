"""
Example script to train a language model using llmbuilder package.
This script demonstrates how to:
1. Load and prepare data
2. Train a tokenizer
3. Train a language model
4. Generate text using the trained model
"""
import os
import sys
from pathlib import Path
from llmbuilder.tokenizer.train import TokenizerTrainer
from llmbuilder.training.train import Trainer as ModelTrainer
from llmbuilder.data.dataset import TextDataset
from llmbuilder.data import split_dataset
from llmbuilder.config import TrainingConfig, ModelConfig

def main():
    # Configuration
    data_path = Path("data/sample_data.txt")  # Path to your training data
    output_dir = Path("output")  # Directory to save the model and outputs
    
    # Clean output directory if it exists
    if output_dir.exists():
        print(f"Cleaning up existing output directory: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
    
    # Create output directories
    model_dir = output_dir / "model"
    tokenizer_dir = output_dir / "tokenizer"
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training with data from: {data_path.absolute()}")
    print(f"Output will be saved to: {output_dir.absolute()}")
    
    try:
        # Step 1: Train tokenizer
        print("\n=== Step 1: Training Tokenizer ===")
        from llmbuilder.config import TokenizerConfig
        
        # Create a custom tokenizer config with a smaller vocab size
        tokenizer_config = TokenizerConfig(
            vocab_size=1000,  # Reduced from default 16000 to 1000
            model_type="bpe",
            character_coverage=1.0,
            max_sentence_length=4096
        )
        
        tokenizer_trainer = TokenizerTrainer(config=tokenizer_config)
        tokenizer_trainer.train(
            input_file=str(data_path),
            output_dir=str(tokenizer_dir),
            model_prefix="tokenizer"
        )
        
        # Step 2: Preprocess and tokenize the text data (using llmbuilder API)
        print("\n=== Step 2: Preprocessing and Tokenizing Data ===")
        
        # Create output directory for tokenized data
        tokenized_data_dir = output_dir / "tokenized_data"
        tokenized_data_dir.mkdir(exist_ok=True)
        tokenized_data_path = tokenized_data_dir / "tokenized_data.pt"
        
        # Use llmbuilder's TokenizerTrainer utility to create a tokenized dataset (.pt)
        dataset_result = tokenizer_trainer.create_tokenized_dataset(
            input_file=str(data_path),
            tokenizer_model=str(tokenizer_dir / "tokenizer.model"),
            output_file=str(tokenized_data_path),
            max_length=None,
        )
        print(f"Tokenized data saved to {dataset_result['output_file']}")
        
        # Step 3: Create dataset from tokenized data
        print("\n=== Step 3: Creating Dataset ===")
        
        # Create dataset from tokenized data
        dataset = TextDataset(
            data_path=str(tokenized_data_path),
            block_size=256,  # Context length
            stride=128,      # Overlap between sequences
            cache_in_memory=True
        )
        
        # Split into train and validation sets (90/10) using llmbuilder API
        # Workaround rounding: provide a tiny non-zero test_ratio to ensure lengths sum correctly
        train_dataset, val_dataset, _ = split_dataset(dataset, train_ratio=0.9, val_ratio=0.1, test_ratio=1e-6)
        
        print(f"Dataset prepared: {len(dataset):,} samples")
        print(f"Training samples: {len(train_dataset):,}, Validation samples: {len(val_dataset):,}")
        
        # Step 3: Train model
        print("\n=== Step 3: Training Model ===")
        
        # Configure training (only supported fields from llmbuilder.config.TrainingConfig)
        config = TrainingConfig(
            batch_size=4,
            num_epochs=3,
            learning_rate=5e-5,
            warmup_steps=100,
            weight_decay=0.01,
            max_grad_norm=1.0,
            save_every=1,   # save checkpoint each epoch
            eval_every=1,   # validate each epoch
            log_every=10
        )
        
        # Initialize trainer with config and a small CPU-friendly model config
        model_config = ModelConfig(
            vocab_size=1000,        # must match tokenizer vocab_size
            embedding_dim=256,
            num_layers=4,
            num_heads=4,
            max_seq_length=256,
            dropout=0.1,
            model_type="gpt"
        )
        trainer = ModelTrainer(config=config, model_config=model_config)
        
        # Train the model
        training_results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_dir=str(output_dir / "checkpoints")
        )
        
        print(f"\nTraining completed. Results: {training_results}")
        
        print("\nTraining completed successfully!")
        
        # Save a proper model checkpoint for inference (includes model_config/metadata)
        from llmbuilder.model import save_model
        inference_model_path = output_dir / "model" / "model.pt"
        save_model(trainer.model, str(inference_model_path))
        
        # Example generation using llmbuilder high-level API
        print("\n=== Example Generation ===")
        from llmbuilder import generate_text
        tok_path = str(tokenizer_dir)
        prompts = [
            "Artificial intelligence is",
            "Machine learning can be used to",
            "The future of AI will"
        ]
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            try:
                generated = generate_text(
                    model_path=str(inference_model_path),
                    tokenizer_path=tok_path,
                    prompt=prompt,
                    max_new_tokens=50,
                    temperature=0.8
                )
                print(f"Generated: {generated}")
            except Exception as ge:
                print(f"Generation failed: {ge}")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
