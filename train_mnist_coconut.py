import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from train_mnist_simple import VisionTransformer

from mnist_coconut import create_mnist_coconut_from_pretrained
from train_mnist_simple import load_mnist_data


def create_reasoning_dataset(images, labels, latent_token_id, start_latent_id, end_latent_id, 
                            num_reasoning_steps=2):
    """Create dataset with reasoning sequences for MNIST classification"""
    dataset = []
    
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        
        # Create reasoning sequence: [start, latent, latent, ..., end]
        reasoning_seq = [start_latent_id]
        for _ in range(num_reasoning_steps):
            reasoning_seq.append(latent_token_id)
        reasoning_seq.append(end_latent_id)
        
        dataset.append({
            'image': image,
            'reasoning_sequence': torch.tensor(reasoning_seq),
            'label': label
        })
    
    return dataset


def collate_reasoning_batch(batch):
    """Collate function for reasoning dataset"""
    images = torch.stack([item['image'] for item in batch])
    reasoning_sequences = torch.stack([item['reasoning_sequence'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'images': images,
        'reasoning_sequences': reasoning_sequences,
        'labels': labels
    }


def train_coconut_model(model_path, data_path, save_path, epochs=5, batch_size=32, lr=1e-4, 
                       num_reasoning_steps=2, use_coconut=True):
    """Train MNIST model with optional Coconut reasoning"""
    
  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if use_coconut:
        # Create Coconut model from pretrained Vision Transformer
        print("Loading pretrained model for Coconut training...")
        coconut_model, config = create_mnist_coconut_from_pretrained(model_path, device)
        model = coconut_model
    else:
        # Load regular Vision Transformer for baseline training
        print("Loading pretrained model for regular training...")
        
        config_path = os.path.join(model_path, "model_config.pth")
        model_config = torch.load(config_path, map_location=device)
        
        model = VisionTransformer(
            img_size=model_config['img_size'],
            patch_size=model_config['patch_size'],
            num_classes=model_config['num_classes'],
            embed_dim=model_config['embed_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads']
        )
        
        model_weights_path = os.path.join(model_path, "mnist_attention_model.pth")
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device)
        
        config = model_config
    
    # Load MNIST data
    print("Loading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_mnist_data(data_path)
    
    if use_coconut:
        # Create reasoning datasets for Coconut training
        print("Creating reasoning datasets...")
        train_reasoning_dataset = create_reasoning_dataset(
            train_images, train_labels, 
            model.latent_token_id, 
            model.start_latent_id, 
            model.end_latent_id,
            num_reasoning_steps
        )
        
        test_reasoning_dataset = create_reasoning_dataset(
            test_images, test_labels,
            model.latent_token_id, 
            model.start_latent_id, 
            model.end_latent_id,
            num_reasoning_steps
        )
        
        # Create data loaders for reasoning
        train_loader = DataLoader(
            train_reasoning_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_reasoning_batch
        )
        
        test_loader = DataLoader(
            test_reasoning_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_reasoning_batch
        )
        
        print(f"Training samples: {len(train_reasoning_dataset)}")
        print(f"Test samples: {len(test_reasoning_dataset)}")
    else:
        # Create regular datasets for standard training
        print("Creating standard datasets...")
        from torch.utils.data import TensorDataset
        
        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    mode_name = "Coconut" if use_coconut else "Regular"
    print(f"Training MNIST {mode_name} for {epochs} epochs...")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        if use_coconut:
            # Coconut training loop
            for batch in pbar:
                images = batch['images'].to(device)
                reasoning_sequences = batch['reasoning_sequences'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with reasoning
                output = model(images, reasoning_sequence=reasoning_sequences, labels=labels)
                loss = output.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Get predictions from the last position
                with torch.no_grad():
                    final_logits = output.logits[:, -1, :]  # Last position predictions
                    pred = final_logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        else:
            # Regular training loop
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluation every epoch
        model.eval()
        eval_correct = 0
        eval_total = 0
        eval_loss = 0
        
        with torch.no_grad():
            if use_coconut:
                # Coconut evaluation
                for batch in test_loader:
                    images = batch['images'].to(device)
                    reasoning_sequences = batch['reasoning_sequences'].to(device)
                    labels = batch['labels'].to(device)
                    
                    output = model(images, reasoning_sequence=reasoning_sequences, labels=labels)
                    eval_loss += output.loss.item()
                    
                    # Get predictions from the last position
                    final_logits = output.logits[:, -1, :]
                    pred = final_logits.argmax(dim=1)
                    eval_correct += (pred == labels).sum().item()
                    eval_total += labels.size(0)
            else:
                # Regular evaluation
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    eval_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    eval_correct += pred.eq(target.view_as(pred)).sum().item()
                    eval_total += target.size(0)
        
        eval_acc = 100. * eval_correct / eval_total
        eval_avg_loss = eval_loss / len(test_loader)
        print(f"Eval: Loss: {eval_avg_loss:.4f}, Acc: {eval_acc:.2f}%")
        
        model.train()
    
    # Final evaluation
    print(f"\nFinal evaluation...")
    model.eval()
    final_correct = 0
    final_total = 0
    
    with torch.no_grad():
        if use_coconut:
            # Test generation mode (no reasoning sequence provided)
            print("Testing generation mode (without explicit reasoning sequence)...")
            for batch in test_loader:
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                
                # Use generation mode
                predictions, _ = model.generate(images, max_reasoning_steps=num_reasoning_steps)
                final_correct += (predictions == labels).sum().item()
                final_total += labels.size(0)
            
            final_acc = 100. * final_correct / final_total
            print(f"Generation mode accuracy: {final_acc:.2f}%")
        else:
            # Regular final evaluation
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                final_correct += pred.eq(target.view_as(pred)).sum().item()
                final_total += target.size(0)
            
            final_acc = 100. * final_correct / final_total
            print(f"Final test accuracy: {final_acc:.2f}%")
    
    # Save the trained model
    os.makedirs(save_path, exist_ok=True)
    
    if use_coconut:
        torch.save(model.state_dict(), os.path.join(save_path, "mnist_coconut_model.pth"))
        
        # Save configuration
        coconut_config = {
            'base_config': config,
            'num_reasoning_steps': num_reasoning_steps,
            'latent_token_id': model.latent_token_id,
            'start_latent_id': model.start_latent_id,
            'end_latent_id': model.end_latent_id,
            'final_accuracy': final_acc,
            'training_mode': 'coconut'
        }
        torch.save(coconut_config, os.path.join(save_path, "coconut_config.pth"))
        print(f"MNIST Coconut model saved to {save_path}")
    else:
        torch.save(model.state_dict(), os.path.join(save_path, "mnist_regular_model.pth"))
        
        # Save configuration
        regular_config = {
            'base_config': config,
            'final_accuracy': final_acc,
            'training_mode': 'regular'
        }
        torch.save(regular_config, os.path.join(save_path, "regular_config.pth"))
        print(f"MNIST Regular model saved to {save_path}")
    
    return model


def test_reasoning_visualization(model, test_loader, device, num_samples=3):
    """Test and visualize reasoning process"""
    model.eval()
    print(f"\nTesting reasoning visualization on {num_samples} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = batch['images'][:1].to(device)  # Take first sample
            labels = batch['labels'][:1].to(device)
            reasoning_sequences = batch['reasoning_sequences'][:1].to(device)
            
            print(f"\nSample {i+1}:")
            print(f"True label: {labels[0].item()}")
            
            # Forward pass with reasoning
            output = model(images, reasoning_sequence=reasoning_sequences, labels=labels)
            
            # Show reasoning at each step
            print("Reasoning steps:")
            for step in range(output.logits.shape[1]):
                step_logits = output.logits[0, step, :]
                step_pred = step_logits.argmax().item()
                step_conf = torch.softmax(step_logits, dim=0)[step_pred].item()
                print(f"  Step {step}: Predicted {step_pred} (confidence: {step_conf:.3f})")
            
            # Test generation mode
            gen_pred, gen_logits = model.generate(images)
            print(f"Generation mode prediction: {gen_pred[0].item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="mnist_attention_checkpoint", 
                       help="Path to pretrained Vision Transformer")
    parser.add_argument("--data_path", default="data", help="Path to MNIST data")
    parser.add_argument("--save_path", default="mnist_coconut_checkpoint", help="Path to save Coconut model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--reasoning_steps", type=int, default=10, help="Number of reasoning steps")
    parser.add_argument("--use_coconut", action="store_true", default=True, help="Use Coconut reasoning (default: True)")
    parser.add_argument("--use_regular", action="store_true", help="Use regular training instead of Coconut")
    
    args = parser.parse_args()
    
    # Handle the toggle logic
    if args.use_regular:
        use_coconut = False
    else:
        use_coconut = args.use_coconut
    
    # Train the model
    trained_model = train_coconut_model(
        model_path=args.model_path,
        data_path=args.data_path,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_reasoning_steps=args.reasoning_steps,
        use_coconut=use_coconut
    )
    
    # Test reasoning visualization (only for Coconut mode)
    if use_coconut:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, test_images, test_labels = load_mnist_data(args.data_path)
        test_reasoning_dataset = create_reasoning_dataset(
            test_images, test_labels,
            trained_model.latent_token_id,
            trained_model.start_latent_id,
            trained_model.end_latent_id,
            args.reasoning_steps
        )
        test_loader = DataLoader(
            test_reasoning_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_reasoning_batch
        )
        
        test_reasoning_visualization(trained_model, test_loader, device)
    else:
        print("Reasoning visualization is only available in Coconut mode.")