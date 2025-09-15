import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from train_mnist_simple import VisionTransformer

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class MNISTCoconut(nn.Module):
    """Coconut architecture adapted for MNIST Vision Transformer"""
    
    def __init__(
        self,
        base_vision_model,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        num_classes=10,
        num_reasoning_layers=2
    ):
        super(MNISTCoconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_vision_model = base_vision_model
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.num_classes = num_classes
        
        # Get embedding dimension from the base model
        self.embed_dim = base_vision_model.patch_embed.projection.out_channels
        
        # Create embeddings for special tokens
        self.special_embeddings = nn.Embedding(3, self.embed_dim)  # latent, start, end
        
        # Learned transformation for updating latent tokens from hidden states
        self.latent_update = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
    def forward(self, images, reasoning_sequence=None, labels=None, **kwargs):
        """
        Forward pass for MNIST Coconut
        
        Args:
            images: (batch_size, channels, height, width) - MNIST images
            reasoning_sequence: (batch_size, seq_len) - sequence with latent tokens for reasoning
            labels: (batch_size,) - target labels for classification
        """
        
        if reasoning_sequence is not None:
            # Training mode with reasoning sequence
            return self._forward_with_reasoning(images, reasoning_sequence, labels)
        else:
            # Inference mode
            return self._forward_classification(images, labels)
    
    def _forward_classification(self, images, labels=None):
        """Standard classification forward pass"""
        logits = self.base_vision_model(images)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return Outputs(loss=loss, inputs_embeds=None, logits=logits)
    
    def _forward_with_reasoning(self, images, reasoning_sequence, labels=None):
        """Forward pass with continuous latent reasoning - hidden states as thoughts"""
        batch_size = images.shape[0]
        
        # Get initial image embeddings
        image_embeds = self._get_image_embeddings(images)  # (batch_size, num_patches+1, embed_dim)
        
        # Initialize reasoning sequence embeddings
        current_sequence_embeds = self._get_sequence_embeddings(reasoning_sequence, image_embeds[:, 0])
        
        # Find latent token positions for continuous reasoning
        latent_indices = (reasoning_sequence == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(batch_size)
        ]
        
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        
        # Initialize growing reasoning sequence with start token
        growing_reasoning_embeds = self._get_sequence_embeddings(
            reasoning_sequence[:, :1], image_embeds[:, 0]  # Only start token initially
        )

        # Continuous autoregressive reasoning loop - GROWING sequence
        for reasoning_step in range(max_n_latents + 1):
            # Create combined input: image patches + GROWING reasoning sequence
            combined_input = torch.cat([image_embeds, growing_reasoning_embeds], dim=1)

            # Apply positional embeddings for current sequence length
            extended_pos_embed = self._get_extended_positional_embeddings(combined_input.shape[1])
            combined_input = combined_input + extended_pos_embed
            combined_input = self.base_vision_model.dropout(combined_input)

            # Forward through base model transformer blocks
            x = combined_input
            for block in self.base_vision_model.blocks:
                x = block(x)
            x = self.base_vision_model.norm(x)

            # Extract reasoning sequence hidden states (continuous thoughts!)
            reasoning_hidden_states = x[:, image_embeds.shape[1]:, :]  # (batch, current_reasoning_len, embed_dim)

            # GROWING AUTOREGRESSIVE: Add next reasoning token to sequence
            if reasoning_step < max_n_latents:
                # Get the next latent token embedding
                next_token_pos = reasoning_step + 1  # +1 because we started with start token
                if next_token_pos < reasoning_sequence.shape[1]:
                    next_token_embeds = self._get_sequence_embeddings(
                        reasoning_sequence[:, next_token_pos:next_token_pos+1],
                        reasoning_hidden_states[:, -1, :]  # Use last hidden state as context
                    )

                    # GROW the sequence by appending next token
                    growing_reasoning_embeds = torch.cat([growing_reasoning_embeds, next_token_embeds], dim=1)
            else:
                # Final step: add end token
                end_token_embeds = self._get_sequence_embeddings(
                    reasoning_sequence[:, -1:], image_embeds[:, 0]  # End token
                )
                growing_reasoning_embeds = torch.cat([growing_reasoning_embeds, end_token_embeds], dim=1)
        
        # Generate final logits from the last reasoning step
        final_logits = self.base_vision_model.head(reasoning_hidden_states)  # (batch, seq_len, num_classes)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Take the last position's logits as the final classification
            final_class_logits = final_logits[:, -1, :]  # (batch_size, num_classes)
            loss = loss_fct(final_class_logits, labels)
        
        return Outputs(loss=loss, inputs_embeds=current_sequence_embeds, logits=final_logits)
    
    def _get_sequence_embeddings(self, reasoning_sequence, image_features):
        """Convert reasoning sequence to embeddings, incorporating image features"""
        batch_size, seq_len = reasoning_sequence.shape
        
        # Initialize embeddings
        sequence_embeds = torch.zeros(batch_size, seq_len, self.embed_dim, device=reasoning_sequence.device)
        
        for i in range(seq_len):
            token_ids = reasoning_sequence[:, i]
            
            # Handle different token types
            for b in range(batch_size):
                token_id = token_ids[b].item()
                
                match token_id:
                    case self.latent_token_id:
                        # Latent token - start with image features
                        sequence_embeds[b, i] = image_features[b]
                    case self.start_latent_id:
                        sequence_embeds[b, i] = self.special_embeddings(torch.tensor(1, device=reasoning_sequence.device))
                    case self.end_latent_id:
                        sequence_embeds[b, i] = self.special_embeddings(torch.tensor(2, device=reasoning_sequence.device))
                    case _:
                        # Should not occur.
                        sequence_embeds[b, i] = self.special_embeddings(torch.tensor(0, device=reasoning_sequence.device))
        
        # Note: Since this place holder is pre-allocated *for each image in batch* 
        # this initial placeholder is used accross the entire
        # batch. i.e. size is [batch_size, seq_len, embed_dim]
        return sequence_embeds
    
    def _get_image_embeddings(self, images):
        """Get image embeddings from base model (up to final hidden states)"""
        # Process image through base model up to hidden states
        x = self.base_vision_model.patch_embed(images)
        
        # Add class token
        cls_tokens = self.base_vision_model.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        return x  # Return raw embeddings before positional embedding
    
    def _get_extended_positional_embeddings(self, total_length):
        """Extend positional embeddings for image + reasoning sequence"""
        # Base model's positional embeddings
        base_pos_embed = self.base_vision_model.pos_embed  # (1, num_patches+1, embed_dim)
        base_length = base_pos_embed.shape[1]

        if total_length <= base_length:
            return base_pos_embed[:, :total_length, :]

        # Need to extend for reasoning tokens
        extra_length = total_length - base_length

        # Create or extend reasoning positional embeddings dynamically
        if not hasattr(self, 'reasoning_pos_embed'):
            self.reasoning_pos_embed = nn.Parameter(
                torch.zeros(1, extra_length, self.embed_dim, device=base_pos_embed.device)
            )
            # Initialize with small random values
            nn.init.normal_(self.reasoning_pos_embed, std=0.02)
        elif self.reasoning_pos_embed.shape[1] < extra_length:
            # Need to extend existing reasoning embeddings
            current_extra = self.reasoning_pos_embed.shape[1]
            additional_length = extra_length - current_extra
            additional_embeds = nn.Parameter(
                torch.zeros(1, additional_length, self.embed_dim, device=base_pos_embed.device)
            )
            nn.init.normal_(additional_embeds, std=0.02)
            self.reasoning_pos_embed = nn.Parameter(
                torch.cat([self.reasoning_pos_embed, additional_embeds], dim=1)
            )

        # Take only what we need for current sequence length
        reasoning_embeds_needed = self.reasoning_pos_embed[:, :extra_length, :]
        extended_pos_embed = torch.cat([base_pos_embed, reasoning_embeds_needed], dim=1)
        return extended_pos_embed
    
    
    def generate(self, images, max_reasoning_steps=6):
        """Generate reasoning sequence for given images using growing autoregressive approach"""
        batch_size = images.shape[0]
        device = images.device

        # For the growing sequence approach, we need to simulate the process
        # without actually calling forward with a full reasoning sequence

        # Get image embeddings
        image_embeds = self._get_image_embeddings(images)

        # Initialize with start token only
        start_sequence = torch.tensor([[self.start_latent_id]] * batch_size, device=device)
        growing_reasoning_embeds = self._get_sequence_embeddings(start_sequence, image_embeds[:, 0])

        all_logits = []

        # Growing autoregressive generation
        for step in range(max_reasoning_steps + 1):
            # Create combined input: image + current growing sequence
            combined_input = torch.cat([image_embeds, growing_reasoning_embeds], dim=1)

            # Apply positional embeddings
            extended_pos_embed = self._get_extended_positional_embeddings(combined_input.shape[1])
            combined_input = combined_input + extended_pos_embed
            combined_input = self.base_vision_model.dropout(combined_input)

            # Forward through transformer
            x = combined_input
            for block in self.base_vision_model.blocks:
                x = block(x)
            x = self.base_vision_model.norm(x)

            # Get reasoning hidden states
            reasoning_hidden_states = x[:, image_embeds.shape[1]:, :]

            # Generate logits for current sequence
            step_logits = self.base_vision_model.head(reasoning_hidden_states)
            all_logits.append(step_logits)

            # Add next token to growing sequence
            if step < max_reasoning_steps:
                # Add latent token
                next_token_sequence = torch.tensor([[self.latent_token_id]] * batch_size, device=device)
                next_token_embeds = self._get_sequence_embeddings(
                    next_token_sequence, reasoning_hidden_states[:, -1, :]
                )
                growing_reasoning_embeds = torch.cat([growing_reasoning_embeds, next_token_embeds], dim=1)
            else:
                # Add end token
                end_token_sequence = torch.tensor([[self.end_latent_id]] * batch_size, device=device)
                end_token_embeds = self._get_sequence_embeddings(
                    end_token_sequence, image_embeds[:, 0]
                )
                growing_reasoning_embeds = torch.cat([growing_reasoning_embeds, end_token_embeds], dim=1)

        # Use final step's logits for prediction
        final_logits = all_logits[-1][:, -1, :]  # Last position of final step
        predicted_classes = torch.argmax(final_logits, dim=1)

        # Return predictions and final step logits
        return predicted_classes, final_logits
    
    def train(self, mode=True):
        self.base_vision_model.train(mode)
        return super().train(mode)
    
    def eval(self):
        self.base_vision_model.eval()
        return super().eval()


def create_mnist_coconut_from_pretrained(model_path, device='cpu', num_reasoning_layers=2):
    """Create MNIST Coconut model from pretrained Vision Transformer"""
    import os
    
    # Load model configuration
    config_path = os.path.join(model_path, "model_config.pth")
    model_config = torch.load(config_path, map_location=device)
    
    # Create base vision model
    base_model = VisionTransformer(
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        num_classes=model_config['num_classes'],
        embed_dim=model_config['embed_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads']
    )
    
    # Load pretrained weights
    model_weights_path = os.path.join(model_path, "mnist_attention_model.pth")
    base_model.load_state_dict(torch.load(model_weights_path, map_location=device))
    
    # Define special token IDs (you can customize these)
    latent_token_id = 10000
    start_latent_id = 10001
    end_latent_id = 10002
    
    # Create Coconut model with proper reasoning layers
    coconut_model = MNISTCoconut(
        base_vision_model=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        num_classes=model_config['num_classes'],
        num_reasoning_layers=num_reasoning_layers
    )
    
    coconut_model.to(device)
    return coconut_model, model_config


def load_trained_coconut_model(model_path, device='cpu'):
    """Load a fully trained MNIST Coconut model from coconut checkpoint"""
    import os

    # Load coconut configuration
    config_path = os.path.join(model_path, "coconut_config.pth")
    coconut_config = torch.load(config_path, map_location=device)

    # Extract base model config
    base_config = coconut_config['base_config']

    # Create base vision model
    base_model = VisionTransformer(
        img_size=base_config['img_size'],
        patch_size=base_config['patch_size'],
        num_classes=base_config['num_classes'],
        embed_dim=base_config['embed_dim'],
        num_layers=base_config['num_layers'],
        num_heads=base_config['num_heads']
    )

    # Create Coconut model with saved token IDs
    coconut_model = MNISTCoconut(
        base_vision_model=base_model,
        latent_token_id=coconut_config['latent_token_id'],
        start_latent_id=coconut_config['start_latent_id'],
        end_latent_id=coconut_config['end_latent_id']
    )

    # Load trained coconut weights
    model_weights_path = os.path.join(model_path, "mnist_coconut_model.pth")
    state_dict = torch.load(model_weights_path, map_location=device)

    # Load with strict=False to handle potential model architecture mismatches
    missing_keys, unexpected_keys = coconut_model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in model: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    coconut_model.to(device)

    return coconut_model, coconut_config


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create MNIST Coconut from pretrained model
    coconut_model, config = create_mnist_coconut_from_pretrained(
        "mnist_attention_checkpoint", 
        device=device
    )
    
    print(f"Created MNIST Coconut model with config: {config}")
    print(f"Model device: {device}")
    
    # Test with dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 1, 28, 28).to(device)
    dummy_labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Test generation
    print("\nTesting generation...")
    with torch.no_grad():
        predictions, reasoning_logits = coconut_model.generate(dummy_images)
        print(f"Predictions: {predictions}")
        print(f"Reasoning logits shape: {reasoning_logits.shape}")
    
    # Test forward pass with reasoning
    print("\nTesting forward pass with reasoning...")
    reasoning_seq = torch.tensor([
        [10001, 10000, 10000, 10002],  # start, latent, latent, end
        [10001, 10000, 10000, 10002]
    ]).to(device)
    
    output = coconut_model(dummy_images, reasoning_sequence=reasoning_seq, labels=dummy_labels)
    print(f"Loss: {output.loss}")
    print(f"Output logits shape: {output.logits.shape}")