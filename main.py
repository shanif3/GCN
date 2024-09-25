import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

# Load the feature matrix from a pickle file
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# Load the edge list from a CSV file (assuming columns are 'source' and 'target')
edges_df = pd.read_csv('edges.csv')

# Step 1: Create a mapping from node names to indices
node_names = pd.concat([edges_df['source'], edges_df['target']]).unique()
name_to_index = {name: i for i, name in enumerate(node_names)}

# Step 2: Replace node names with indices in the edges DataFrame
edges_df['source'] = edges_df['source'].map(name_to_index)
edges_df['target'] = edges_df['target'].map(name_to_index)

# Step 3: Convert edge list to edge index tensor
edges = edges_df[['source', 'target']].values
edge_index = torch.tensor(edges.T, dtype=torch.long)  # shape [2, num_edges]

# Load labels for node classification (replace with your actual data)
num_nodes = features.shape[0]
labels = features['label'].values
features = features.drop(columns='label').values

# Create a PyTorch Geometric data object
data = Data(
    x=torch.tensor(features, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(labels, dtype=torch.long)
)

# Create train/validation/test masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

indices = np.arange(num_nodes)
np.random.shuffle(indices)
train_size = int(0.6 * num_nodes)
val_size = int(0.2 * num_nodes)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


# Define the GCN model using PyTorch Lightning
class GCN(pl.LightningModule):
    def __init__(self, num_features, data):
        super(GCN, self).__init__()
        self.data = data  # Store the data as an attribute
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 1)  # Binary classification: 1 output node

    def forward(self):
        x, edge_index = self.data.x, self.data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # Raw logits for binary classification

    def training_step(self, batch, batch_idx):
        out = self.forward()
        loss = F.binary_cross_entropy_with_logits(out[self.data.train_mask], self.data.y[self.data.train_mask].float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward()
        val_loss = F.binary_cross_entropy_with_logits(out[self.data.val_mask], self.data.y[self.data.val_mask].float())
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        out = self.forward()
        test_loss = F.binary_cross_entropy_with_logits(out[self.data.test_mask],
                                                       self.data.y[self.data.test_mask].float())
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def train_dataloader(self):
        # Return a dummy DataLoader since we use the full graph in each step
        return DataLoader([0], batch_size=1)

    def val_dataloader(self):
        return DataLoader([0], batch_size=1)

    def test_dataloader(self):
        return DataLoader([0], batch_size=1)


# Instantiate the model
model = GCN(num_features=features.shape[1], data=data)

logger = TensorBoardLogger("gcn", name="cnn", version=f"try")

# Train the model
trainer = pl.Trainer(max_epochs=50, logger=logger)
trainer.fit(model)

# Save the model after training
trainer.save_checkpoint("gcn_model_checkpoint.ckpt")

# Load the model later for evaluation
model = GCN.load_from_checkpoint("gcn_model_checkpoint.ckpt", num_features=features.shape[1], data=data)


# After training, evaluate with a new node
# Function to evaluate a new node with new edges
def evaluate_new_node(model, new_node_name, new_node_features, new_edges_df):
    # Check if the new node is already in the graph
    if new_node_name not in name_to_index:
        # Step 1: Add the new node's features to the feature matrix
        data.x = torch.cat([data.x, new_node_features], dim=0)

        # Step 2: Update the name_to_index dictionary with the new node
        new_node_index = data.x.shape[0] - 1  # Index of the new node
        name_to_index[new_node_name] = new_node_index
    else:
        # If the node already exists, retrieve its index
        new_node_index = name_to_index[new_node_name]

    # Map the node names in new_edges_df to indices using the updated name_to_index
    new_edges_df['source'] = new_edges_df['source'].map(name_to_index)
    new_edges_df['target'] = new_edges_df['target'].map(name_to_index)

    # Create the new_edges tensor
    new_edges = torch.tensor(new_edges_df.values.T, dtype=torch.long)  # shape [2, num_new_edges]

    # Append the new edges to the existing edge_index
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    # Set the model to evaluation mode
    model.eval()

    # Perform a forward pass
    with torch.no_grad():
        out = model()

    # Get the prediction for the new node
    new_node_output = out[new_node_index]
    predicted_label = torch.sigmoid(new_node_output).item()

    # Return the binary classification result
    return 1 if predicted_label >= 0.5 else 0


# Example usage: adding a new node and evaluating it
new_node_name = 'new_node_1'
new_node_features = torch.tensor(np.random.randn(1, features.shape[1]), dtype=torch.float)

# Example new edges involving the new node
new_edges_df = pd.DataFrame({
    'source': [new_node_name, new_node_name],
    'target': ['existing_node_1', 'existing_node_2']  # Replace with actual node names
})

# Evaluate the new node
predicted_label = evaluate_new_node(model, new_node_name, new_node_features, new_edges_df)
print(f"The predicted label for the new node is: {predicted_label}")