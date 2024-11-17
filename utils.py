import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from skimage.segmentation import mark_boundaries

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

import shap

from lime import lime_image

def get_class_names_from_idxs(idx_list, class_to_idx_map):
    """
    Converts a list of class indices into their corresponding class names.

    This function takes a list, single integer, or a PyTorch tensor of class indices, 
    and returns a list of class names based on the class_to_idx_map global mapping.

    Args:
        idx_list: List of class indices, a single integer index, or a PyTorch tensor 
                  containing class indices.
        class_to_idx_map: A dictionary mapping class names to their respective indices.

    Returns:
        List of class names corresponding to the given indices.
    """
    # Reverse the class_to_idx_map to get an index-to-class mapping
    idx_to_class_map = {v: k for k, v in class_to_idx_map.items()}

    # Handle single integer or numpy int64 inputs by converting them to a list
    if type(idx_list) == int or type(idx_list) == np.int64:
        idx_list = [idx_list]
    
    # Handle PyTorch tensor inputs by converting them to a list of integers
    if type(idx_list) == torch.Tensor:
        if idx_list.dim() == 0:  # Scalar tensor
            idx_list = [int(idx_list.item())]
        else:  # Tensor with multiple values
            idx_list = [int(t.item()) for t in idx_list]
    
    # Map indices to class names
    return [idx_to_class_map[idx] for idx in idx_list]


def plot_class_distribution(train_ds, val_ds, test_ds, class_names):
    """
    Plots the class distribution as percentages across training, testing, and validation datasets.

    This function calculates the percentage distribution of each class in the given datasets 
    (train, test, and validation) and visualizes the results in a bar chart. The chart shows 
    each class as a group of three bars, representing the train, test, and validation percentages.

    Args:
        train_ds: Iterable dataset for training. Each item is a tuple of (data, label).
        val_ds: Iterable dataset for validation. Each item is a tuple of (data, label).
        test_ds: Iterable dataset for testing. Each item is a tuple of (data, label).
        class_names: List of class names corresponding to the classes.

    Returns:
        None. Displays a bar chart.
    """
    classes = list(range(len(class_names)))
    
    # Extract labels
    train_labels = [label for _, label in train_ds]
    val_labels = [label for _, label in val_ds]
    test_labels = [label for _, label in test_ds]

    # Count labels
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    # Calculate percentages
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())
    
    train_percentages = {cls: (train_counts[cls] / total_train) * 100 for cls in classes}
    val_percentages = {cls: (val_counts[cls] / total_val) * 100 for cls in classes}
    test_percentages = {cls: (test_counts[cls] / total_test) * 100 for cls in classes}

    # Plot
    x = range(len(class_names))
    width = 0.25
    train_bars = plt.bar(x, [train_percentages.get(cls, 0) for cls in classes], width=width, label="Train", align="center")
    val_bars = plt.bar([p + 2 * width for p in x], [val_percentages.get(cls, 0) for cls in classes], width=width, label="Val", align="center")
    test_bars = plt.bar([p + width for p in x], [test_percentages.get(cls, 0) for cls in classes], width=width, label="Test", align="center")

    # Add values above bars
    def add_values(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0, 
                height + 0.5, 
                f'{height:.1f}%', 
                ha='center', 
                va='bottom', 
                fontsize=6, 
                rotation=45
            )
    add_values(train_bars)
    add_values(val_bars)
    add_values(test_bars)

    max_height = max(
        max([bar.get_height() for bar in train_bars]),
        max([bar.get_height() for bar in val_bars]),
        max([bar.get_height() for bar in test_bars])
    )
    
    plt.ylim(0, max_height + 5)
    plt.xlabel("Classes")
    plt.ylabel("Percentage Count (%)")
    plt.title("Class Distribution Across Train, Val and Test Sets")
    plt.xticks([p + width for p in x], class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) architecture for image classification.

    This model consists of two main sections:
    1. Convolutional Section: Extracts spatial features from input images using convolutional layers,
       max pooling, batch normalization, and ReLU activations.
    2. Fully Connected (FC) Section: Maps the flattened features to class scores using linear layers 
       and ReLU activations.

    Attributes:
        conv_section (nn.Sequential): The convolutional section of the model, consisting of:
            - Conv2d: First convolutional layer (3 input channels, 8 output channels, kernel size 5).
            - MaxPool2d: Downsamples by a factor of 2.
            - BatchNorm2d: Normalizes feature maps with 8 channels.
            - ReLU: Activation function.
            - Conv2d: Second convolutional layer (8 input channels, 16 output channels, kernel size 3).
            - MaxPool2d: Downsamples by a factor of 2.
            - BatchNorm2d: Normalizes feature maps with 16 channels.
            - ReLU: Activation function.
        
        fc_section (nn.Sequential): The fully connected section of the model, consisting of:
            - Linear: Maps 16 * 54 * 54 features to 120 hidden units.
            - ReLU: Activation function.
            - Linear: Maps 120 hidden units to 84 hidden units.
            - ReLU: Activation function.
            - Linear: Maps 84 hidden units to N_CLASSES (number of output classes).

    Methods:
        forward(x): Performs a forward pass through the network, from input images to class scores.

    Note:
        - The model assumes the input image size is (3, 224, 224) to match the dimensions in the FC layer.
    """
    def __init__(self, n_classes: int):
        super(SimpleCNN, self).__init__()
        # Define the convolutional section of the model
        self.conv_section = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5),    # First convolutional layer
            nn.MaxPool2d(2),                  # Downsample by 2
            nn.BatchNorm2d(8),                # Batch normalization for 8 channels
            nn.ReLU(),                        # Activation
            nn.Conv2d(8, 16, kernel_size=3),  # Second convolutional layer
            nn.MaxPool2d(2),                  # Downsample by 2
            nn.BatchNorm2d(16),               # Batch normalization for 16 channels
            nn.ReLU()                         # Activation
        )
        # Define the fully connected section of the model
        self.fc_section = nn.Sequential(
            nn.Linear(16 * 54 * 54, 120),     # Fully connected layer 1
            nn.ReLU(),                        # Activation
            nn.Linear(120, 84),               # Fully connected layer 2
            nn.ReLU(),                        # Activation
            nn.Linear(84, n_classes)          # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, N_CLASSES), 
                          containing class scores for each input image.
        """
        # Pass input through the convolutional section
        conv_out = self.conv_section(x)
        
        # Flatten the output for the fully connected layer
        flatten_out = conv_out.reshape(x.size(0), -1)  # Optimize for batch size
        
        # Pass through the fully connected section
        return self.fc_section(flatten_out)


class RobustCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) architecture for image classification.

    This model is designed with the following sections:
    1. Convolutional Section:
       - Three blocks of convolutional layers, each followed by batch normalization, 
         ReLU activation, and max-pooling for downsampling.
    2. Fully Connected Section:
       - A sequence of linear layers with ReLU activations and dropout for regularization.
       - Maps the flattened output of the convolutional section to the class scores.

    Attributes:
        conv_section (nn.Sequential): Sequential model for feature extraction:
            - Block 1: 2 Conv2d layers (3 -> 16), BatchNorm, ReLU, MaxPool2d.
            - Block 2: 2 Conv2d layers (16 -> 32), BatchNorm, ReLU, MaxPool2d.
            - Block 3: 2 Conv2d layers (32 -> 64), BatchNorm, ReLU, MaxPool2d.
        fc_section (nn.Sequential): Fully connected layers for classification:
            - Linear layers with dimensions dynamically calculated, followed by ReLU and Dropout.
            - Final layer outputs the number of classes (N_CLASSES).
        flatten_dim (int): Flattened dimension of the convolutional output, computed dynamically.

    Methods:
        forward(x): Performs a forward pass through the model.
    """
    def __init__(self, n_classes: int):
        super(RobustCNN, self).__init__()
        
        # Convolutional Section: Feature extraction through convolutional layers
        self.conv_section = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # First convolutional layer
            nn.BatchNorm2d(16),                          # Batch normalization for stability
            nn.ReLU(),                                   # Activation function
            nn.Conv2d(16, 16, kernel_size=3, padding=1), # Second convolutional layer
            nn.BatchNorm2d(16),                          # Batch normalization
            nn.ReLU(),                                   # Activation
            nn.MaxPool2d(2),                             # Downsample by 2

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # First convolutional layer
            nn.BatchNorm2d(32),                          # Batch normalization
            nn.ReLU(),                                   # Activation
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # Second convolutional layer
            nn.BatchNorm2d(32),                          # Batch normalization
            nn.ReLU(),                                   # Activation
            nn.MaxPool2d(2),                             # Downsample by 2

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # First convolutional layer
            nn.BatchNorm2d(64),                          # Batch normalization
            nn.ReLU(),                                   # Activation
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Second convolutional layer
            nn.BatchNorm2d(64),                          # Batch normalization
            nn.ReLU(),                                   # Activation
            nn.MaxPool2d(2),                             # Downsample by 2
        )
        
        # Dynamically calculate the flatten dimension after the convolutional section
        self.flatten_dim = self._get_flatten_dim()
        
        # Fully Connected Section: Maps extracted features to class scores
        self.fc_section = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),  # First linear layer
            nn.ReLU(),                         # Activation
            nn.Dropout(0.5),                   # Dropout for regularization
            nn.Linear(256, 128),               # Second linear layer
            nn.ReLU(),                         # Activation
            nn.Dropout(0.5),                   # Dropout for regularization
            nn.Linear(128, n_classes),         # Output layer
        )

    def _get_flatten_dim(self) -> int:
        """
        Computes the dimension of the flattened tensor after the convolutional section.

        This method passes a dummy input tensor of size (1, 3, 224, 224) through the
        convolutional layers to determine the output size, which is then used as the 
        input size for the fully connected layers.

        Returns:
            int: Flattened dimension of the convolutional output.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # Assuming input image size is 224x224
            conv_out = self.conv_section(dummy_input)
            return int(torch.prod(torch.tensor(conv_out.shape[1:])))  # Product of all dimensions except batch size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224), where
                              batch_size is the number of input samples.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, N_CLASSES), containing
                          class scores for each input sample.
        """
        # Pass input through the convolutional section
        x = self.conv_section(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, flatten_dim)
        
        # Pass through the fully connected section
        x = self.fc_section(x)
        return x


def train_model_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using the provided training data loader.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function to compute training loss.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the computations (e.g., "cpu" or "cuda").

    Returns:
        tuple: A tuple containing the following metrics:
            - avg_loss (float): Average training loss over all batches.
            - accuracy (float): Overall training accuracy.
            - precision (float): Weighted average precision across all classes.
            - recall (float): Weighted average recall across all classes.
            - f1 (float): Weighted average F1 score across all classes.

    Notes:
        - This function performs forward and backward passes for each batch, updates 
          the model's parameters using the optimizer, and computes various training metrics.
    """
    model.train()  # Set the model to training mode
    total_train, correct_train, train_loss = 0, 0, 0.0
    all_train_preds, all_train_labels = [], []

    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass: compute predictions and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backpropagation and optimization step
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        # Track accuracy and loss
        _, preds = torch.max(outputs, 1)  # Get the predicted class indices
        correct_train += preds.eq(targets).sum().item()  # Count correct predictions
        total_train += targets.size(0)  # Total number of samples
        train_loss += loss.item()  # Accumulate loss

        # Collect predictions and true labels for metric calculation
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(targets.cpu().numpy())

    # Calculate metrics
    accuracy = correct_train / total_train
    precision = precision_score(all_train_labels, all_train_preds, average="weighted")
    recall = recall_score(all_train_labels, all_train_preds, average="weighted")
    f1 = f1_score(all_train_labels, all_train_preds, average="weighted")
    avg_loss = train_loss / len(train_loader)

    return avg_loss, accuracy, precision, recall, f1


def evaluate_model(model, val_loader, criterion, device, class_names, print_cf_report=False):
    """
    Evaluates the model using the provided validation data loader.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function to compute validation loss.
        device (torch.device): Device to run the computations (e.g., "cpu" or "cuda").
        class_names: List of class names corresponding to the classes.
        print_cf_report (bool): If True, prints a classification report.

    Returns:
        tuple: A tuple containing the following metrics:
            - avg_loss (float): Average validation loss over all batches.
            - accuracy (float): Overall validation accuracy.
            - precision (float): Weighted average precision across all classes.
            - recall (float): Weighted average recall across all classes.
            - f1 (float): Weighted average F1 score across all classes.

    Notes:
        - This function does not perform backpropagation or update the model's parameters.
        - It operates in evaluation mode, disabling gradient computation for efficiency.
        - The classification report (if enabled) provides detailed per-class performance metrics.
    """
    model.eval()  # Set the model to evaluation mode
    total_val, correct_val, val_loss = 0, 0, 0.0
    all_val_preds, all_val_labels = [], []

    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass: compute predictions and loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Track accuracy and loss
            _, preds = torch.max(outputs, 1)  # Get the predicted class indices
            correct_val += preds.eq(targets).sum().item()  # Count correct predictions
            total_val += targets.size(0)  # Total number of samples
            val_loss += loss.item()  # Accumulate loss

            # Collect predictions and true labels for metric calculation
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(targets.cpu().numpy())

    # Calculate metrics
    accuracy = correct_val / total_val
    precision = precision_score(all_val_labels, all_val_preds, average="weighted")
    recall = recall_score(all_val_labels, all_val_preds, average="weighted")
    f1 = f1_score(all_val_labels, all_val_preds, average="weighted")
    avg_loss = val_loss / len(val_loader)

    if print_cf_report:
        print(classification_report(all_val_labels, all_val_preds, target_names=class_names))

    return avg_loss, accuracy, precision, recall, f1


def train_model(model, train_loader, val_loader, criterion, optimizer, device, class_names, model_save_path, epochs=30, patience=3):
    """
    Trains the model using the provided training and validation datasets with early stopping.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function to compute training and validation loss.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the computations (e.g., "cpu" or "cuda").
        class_names: List of class names corresponding to the classes.
        model_save_path (str): Path to save the trained model's state_dict.
        epochs (int, optional): Number of training epochs. Default is 30.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is 3.

    Returns:
        Training and validation metrics as dict. The trained model's state_dict is saved to the specified path.
    """
    best_val_loss = float('inf')  # Initialize the best validation loss
    early_stop_counter = 0       # Counter to track early stopping
    train_metrics = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []
    }
    val_metrics = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []
    }

    print("==> Training starts!")
    print("=" * 50)

    # Training loop with validation and early stopping
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Train for one epoch
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_model_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")

        # Record training metrics
        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(train_accuracy)
        train_metrics['precision'].append(train_precision)
        train_metrics['recall'].append(train_recall)
        train_metrics['f1_score'].append(train_f1)
        
        # Validate on validation set
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, criterion, device, class_names
        )
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}\n")

        # Record validation metrics
        val_metrics['loss'].append(val_loss)
        val_metrics['accuracy'].append(val_accuracy)
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['f1_score'].append(val_f1)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0  # Reset counter if there is an improvement
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience} \n")

        if early_stop_counter >= patience:
            print("Early stopping triggered. \n")
            break

    print("==> Training complete!")

    # Save the trained model's state_dict
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved at: {model_save_path}")

    return train_metrics, val_metrics


import matplotlib.pyplot as plt

def plot_training_metrics(train_metrics, val_metrics):
    """
    Plots the training and validation metrics (accuracy, loss, precision, recall, and F1 score) over epochs.

    Args:
        train_metrics (dict): A dictionary containing training metrics with keys:
            - 'accuracy': List of accuracy values for each epoch.
            - 'loss': List of loss values for each epoch.
            - 'precision': List of precision values for each epoch.
            - 'recall': List of recall values for each epoch.
            - 'f1_score': List of F1 score values for each epoch.
        val_metrics (dict): A dictionary containing validation metrics with the same keys as `train_metrics`.

    Returns:
        None. Displays the plots for training and validation metrics.
    """
    # Define epochs based on the number of recorded metrics
    epochs = range(1, len(train_metrics['accuracy']) + 1)

    # Create a figure for the plots with a 3x2 grid (6 plots total)
    plt.figure(figsize=(15, 10))

    # Plot accuracy
    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_metrics['accuracy'], label='Train')
    plt.plot(epochs, val_metrics['accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_metrics['loss'], label='Train')
    plt.plot(epochs, val_metrics['loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot precision
    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_metrics['precision'], label='Train')
    plt.plot(epochs, val_metrics['precision'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()

    # Plot recall
    plt.subplot(3, 2, 4)
    plt.plot(epochs, train_metrics['recall'], label='Train')
    plt.plot(epochs, val_metrics['recall'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall')
    plt.legend()

    # Plot F1 score
    plt.subplot(3, 2, 5)
    plt.plot(epochs, train_metrics['f1_score'], label='Train')
    plt.plot(epochs, val_metrics['f1_score'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()

    # Adjust layout for better visualization
    plt.tight_layout()

    # Show the plots
    plt.show()


def convert_tensor_to_np(tensor_image):
    """
    Converts a PyTorch tensor image to a NumPy array.

    This function ensures the input tensor is converted to a NumPy array and 
    rearranges the dimensions from channel-first (CHW) to channel-last (HWC) format, 
    suitable for visualization or further processing.

    Args:
        tensor_image (torch.Tensor): A PyTorch tensor representing the image, 
                                     with dimensions (C, H, W).

    Returns:
        np.ndarray: A NumPy array representation of the image, 
                    with dimensions (H, W, C).
    """
    np_image = tensor_image

    # Check if the input is not already a NumPy array
    if type(np_image) != np.ndarray:
        # Move the tensor to the CPU and convert to a NumPy array
        np_image = np_image.cpu().numpy()
        # Rearrange dimensions from CHW (channel-first) to HWC (channel-last)
        np_image = np.transpose(np_image, (1, 2, 0))

    return np_image


def convert_np_to_tensor(np_image, device):
    """
    Converts a NumPy array image to a PyTorch tensor.

    Args:
        np_image (np.ndarray): A NumPy array representing the image, 
                               with dimensions (H, W, C) for a single image 
                               or (N, H, W, C) for a batch.
        device (torch.device): Device to run the computations (e.g., "cpu" or "cuda").

    Returns:
        torch.Tensor: A PyTorch tensor representation of the image, 
                      with dimensions (C, H, W) for a single image 
                      or (N, C, H, W) for a batch.
    """
    # Convert NumPy array to PyTorch tensor
    if not isinstance(np_image, torch.Tensor):
        tensor_image = torch.from_numpy(np_image)
    else:
        tensor_image = np_image

    # If the input is a single image (H, W, C)
    if tensor_image.dim() == 3:  # Single image
        tensor_image = tensor_image.permute(2, 0, 1)  # Convert HWC to CHW

    # If the input is a batch of images (N, H, W, C)
    elif tensor_image.dim() == 4:  # Batch of images
        tensor_image = tensor_image.permute(0, 3, 1, 2)  # Convert NHWC to NCHW

    # Convert to float and move to the specified device
    tensor_image = tensor_image.float().to(device)

    return tensor_image


def model_predict(model, input_img, device):
    """
    Predicts the class probabilities for a given input image.

    Args:
        model (torch.nn.Module): The neural network model to be used for prediction.
        input_img (torch.Tensor or np.ndarray): Input image in (H, W, C) or (C, H, W) format.
        device (torch.device): Device to run computations (e.g., "cpu" or "cuda").

    Returns:
        np.ndarray: An array of probabilities for each class.
    """
    # Ensure the input image is a PyTorch tensor
    if not isinstance(input_img, torch.Tensor):
        input_img = torch.from_numpy(input_img)

    # If the input image is in HWC format, convert it to CHW
    if input_img.dim() == 3 and input_img.shape[-1] in {1, 3}:  # H, W, C format
        input_img = input_img.permute(2, 0, 1)  # Convert to C, H, W

    # Add a batch dimension if missing
    if input_img.dim() == 3:
        input_img = input_img.unsqueeze(0)  # Convert to N, C, H, W

    # Move the image to the correct device
    input_tensor = input_img.to(device)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Pass the input tensor through the model
        output = model(input_tensor)
        # Apply softmax to convert raw scores to probabilities
        probas = torch.nn.functional.softmax(output, dim=1)
    return probas.cpu().numpy()


def lime_explain_image(model, image_dict, n_classes, device):
    """
    Generates and visualizes a LIME (Local Interpretable Model-Agnostic Explanations) explanation 
    for an image classification prediction.

    Args:
        model (torch.nn.Module): The neural network model to be used for prediction.
        image_dict (dict): A dictionary containing:
            - 'image' (torch.Tensor): The input image tensor in (C, H, W) format.
            - 'label' (str): The true label of the image.
            - 'pred_label' (str): The predicted label of the image.
        n_classes (int): The number of classes.
        device (torch.device): The device to run computations (e.g., "cpu" or "cuda").

    Returns:
        explanation: A LIME explanation object containing the details of the superpixel contributions.
    """
    # Convert the input tensor image to a NumPy array for visualization
    lime_image_np = convert_tensor_to_np(image_dict['image'])

    # Display the original image with true and predicted labels
    plt.imshow(lime_image_np)
    plt.title(f"Label: {image_dict['label']}, Predicted: {image_dict['pred_label']}")
    plt.axis('off')
    plt.show()

    # Set the model to evaluation mode
    model.eval()

    # Define a classifier function compatible with LIME
    def wrapped_model_predict(images):
        """
        Wrapper for the predict function to be compatible with LIME.
        Converts the input images to PyTorch tensors and returns class probabilities.
    
        Args:
            images (list): List of images as NumPy arrays in (H, W, C) format.
    
        Returns:
            np.ndarray: Class probabilities for each input image.
        """
        tensor_images = []
        for img in images:
            tensor_img = convert_np_to_tensor(img, device)  # Convert to tensor (C, H, W)
            tensor_images.append(tensor_img)
    
        # Stack into a batch (N, C, H, W)
        tensor_images = torch.stack(tensor_images, dim=0)
    
        with torch.no_grad():
            outputs = model(tensor_images)
            probas = torch.nn.functional.softmax(outputs, dim=1)
        return probas.cpu().numpy()

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation using LIME
    explanation = explainer.explain_instance(
        lime_image_np,
        wrapped_model_predict,
        top_labels=n_classes,  # Specify the number of top classes to explain
        hide_color=0,          # Color to hide other superpixels
        num_samples=1000       # Number of perturbations for LIME
    )

    # Create a figure with subplots for visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Visualization 1: Positive contributions only
    image, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    axes[0, 0].imshow(mark_boundaries(image / 2 + 0.5, mask))
    axes[0, 0].axis('off')
    axes[0, 0].set_title("LIME Explanation (Positive Only)")

    # Visualization 2: Positive and negative contributions
    image, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    axes[0, 1].imshow(mark_boundaries(image / 2 + 0.5, mask))
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Positive and Negative Contributions")

    # Visualization 3: Superpixel importance heatmap
    heatmap = np.zeros(mask.shape)
    for feature, weight in explanation.local_exp[explanation.top_labels[0]]:
        heatmap[mask == feature] = weight
    axes[1, 0].imshow(image / 2 + 0.5)
    axes[1, 0].imshow(heatmap, cmap=cm.jet, alpha=0.6)  # Overlay heatmap
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Superpixel Importance Heatmap")

    # Visualization 4: Top 3 contributing superpixels
    image, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        num_features=3,
        hide_rest=True
    )
    axes[1, 1].imshow(mark_boundaries(image / 2 + 0.5, mask))
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Top 3 Contributing Superpixels")

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()

    return explanation


def shap_explain_images(shap_explainer, class_names, images, labels):
    """
    Generates SHAP explanations for a subset of images and visualizes the explanations.

    Args:
        shap_explainer (shap.DeepExplainer): SHAP DeepExplainer initialized with the model and background.
        class_names (list): List of class names corresponding to the model's output classes.
        shap_images (list): Sample images to explain.
        shape_labels (list): Labels for the sample images.

    Returns:
        None. Displays SHAP explanation plots using Matplotlib.

    Notes:
        - SHAP explanations visualize how each pixel contributes to the model's predictions.
        - Background samples are used as a baseline for SHAP value computation.
    """
    # Compute SHAP values using the DeepExplainer
    shap_values = shap_explainer.shap_values(images)

    # Convert SHAP values to NumPy format for visualization
    shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))
    test_numpy = np.swapaxes(
        np.swapaxes(
            np.array([img.numpy() for img in images]), 1, -1
        ), 
        1, 2
    )

    # Prepare labels for visualization
    labels_array = np.array([class_names] * len(labels))

    # Visualize SHAP explanations
    shap.plots.image(shap_numpy, test_numpy, labels=labels_array, true_labels=labels)


class GradCAM:
    """
    Implements the GradCAM (Gradient-weighted Class Activation Mapping) 
    technique for visualizing important regions in an image that contributed 
    to a model's prediction.

    Attributes:
        model (torch.nn.Module): The neural network model.
        target_layer (str): The name of the target layer for which activations and gradients 
                            will be captured.
        gradients (torch.Tensor): Gradients of the target layer during backpropagation.
        activations (torch.Tensor): Activations (feature maps) of the target layer.

    Methods:
        hook_layers(): Attaches hooks to capture gradients and activations of the target layer.
        generate_heatmap(input_image, class_idx=None): Generates the GradCAM heatmap for a 
                                                       specified class or the predicted class.
    """
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM with the model and the target layer.

        Args:
            model (torch.nn.Module): The neural network model.
            target_layer (str): The name of the layer to analyze with GradCAM.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        """
        Attach hooks to the target layer to capture activations during the forward pass 
        and gradients during the backward pass.
        """
        def forward_hook(module, input, output):
            # Store the activations (feature maps) of the target layer
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Store the gradients of the target layer during backpropagation
            self.gradients = grad_output[0].detach()

        # Locate the target layer and attach hooks
        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)  # Updated hook method for full backward compatibility

    def generate_heatmap(self, input_image, device, class_idx=None):
        """
        Generates a GradCAM heatmap for a specified class or the predicted class.

        Args:
            input_image (torch.Tensor): Input image tensor in (C, H, W) format.
            device (torch.device): Device to run the computations (e.g., "cpu" or "cuda").
            class_idx (int, optional): Index of the class to generate the heatmap for. 
                                       If None, the predicted class is used.

        Returns:
            np.ndarray: A normalized heatmap highlighting important regions 
                        (values between 0 and 1).
        """
        # Add batch dimension and move to the specified device
        input_image = input_image.unsqueeze(0).to(device)
        
        # Zero gradients in the model
        self.model.zero_grad()

        # Perform forward pass to get the output
        output = self.model(input_image)

        # Determine the class index if not provided
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Compute the gradient of the score for the target class
        score = output[:, class_idx]
        score.backward()

        # Compute weights for the GradCAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Generate the GradCAM heatmap
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)  # Remove negative values
        heatmap /= np.max(heatmap)  # Normalize to [0, 1]

        return heatmap


def gradcam_explain_image(image_dict, heatmap, alpha=0.4):
    """
    Visualizes a GradCAM heatmap overlaid on the input image.

    Args:
        image_dict (dict): Dictionary containing:
            - 'image' (torch.Tensor or np.ndarray): Input image in (C, H, W) format.
            - 'label' (str): True label of the image.
            - 'pred_label' (str): Predicted label of the image.
        heatmap (np.ndarray): GradCAM heatmap as a 2D array with values normalized to [0, 1].
        alpha (float): Weight for blending the heatmap and the original image. 
                       Default is 0.4 (40% heatmap, 60% image).

    Returns:
        None. Displays the overlaid image with heatmap using Matplotlib.

    Notes:
        - The heatmap is resized to match the dimensions of the input image.
        - The function blends the heatmap and the image for better visualization.
    """
    image = image_dict['image']
    
    # Normalize heatmap to [0, 255] range
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap (jet) to the heatmap
    colormap = plt.cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap / 255.0)[:, :, :3]  # Shape: (H, W, 3)
    
    # Convert colormap to tensor and adjust dimensions to (C, H, W)
    heatmap_tensor = torch.from_numpy(heatmap_colored).permute(2, 0, 1).unsqueeze(0).float()
    
    # Resize the heatmap to match the original image dimensions
    resize = transforms.Resize((image.shape[1], image.shape[2]))  # (H, W)
    heatmap_resized = resize(heatmap_tensor)
    
    # Remove batch dimension and convert to NumPy
    heatmap_resized = heatmap_resized.squeeze(0).numpy()
    
    # Normalize the input image to [0, 1] range if needed
    if image.max() > 1:
        image = image / 255.0

    # Convert the image to a NumPy array
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = image

    # Ensure image is in (C, H, W) format
    if image_np.shape[0] != 3:
        image_np = image_np.transpose(2, 0, 1)

    # Overlay the heatmap on the original image
    overlay = alpha * heatmap_resized + (1 - alpha) * image_np

    # Transpose to (H, W, C) format for visualization
    overlay = np.clip(overlay.transpose(1, 2, 0), 0, 1)
    
    plt.imshow(overlay)
    plt.title(f"Label: {image_dict['label']}, Predicted: {image_dict['pred_label']}")
    plt.axis('off')
    plt.show()
