import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
T = 5  # 预测编码层级深度
embed_dim = 10 # Target embed_dim for R[l] and P[l]
cnn_feature_dim = 128 # Output of CNN
hidden_dim = 256 # Hidden dim for MLPs
batch_size = 128
lr = 0.001
epochs = 50 # Set to a lower value like 1-2 for quick testing, 50 for actual training
K = 10 # Number of inference steps
lr_inference_bu = 0.1 # Learning rate for bottom-up inference
lr_inference_td = 0.1 # Learning rate for top-down inference
lambda_bu_predictive = 0.5 # Weight for bottom-up predictive loss
lambda_td_predictive = 0.5 # Weight for top-down predictive loss


# Original PredictiveMLP
class PredictiveMLP(nn.Module):
    def __init__(self, cnn_feature_dim=128, embed_dim=10, hidden_dim=256):
        super().__init__()
        input_dim = cnn_feature_dim + embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x_features, prior_state):
        combined = torch.cat([x_features, prior_state], dim=1)
        return self.mlp(combined)

# Original CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128)
        )
    def forward(self, x):
        return self.features(x)

# New ProjectionLayer
class ProjectionLayer(nn.Module):
    def __init__(self, cnn_feature_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(cnn_feature_dim, embed_dim)
    def forward(self, x_features):
        return self.linear(x_features)

# Modified model initialization 
# Using consistent names for global model variables
cnn_model = CNN()
projection_layer_model = ProjectionLayer(cnn_feature_dim, embed_dim)
bottom_up_mlps_models = nn.ModuleList([PredictiveMLP(cnn_feature_dim, embed_dim, hidden_dim) for _ in range(T)])
feedback_mlps_models = nn.ModuleList([PredictiveMLP(cnn_feature_dim, embed_dim, hidden_dim) for _ in range(T - 1)])

# Optimizer instantiation
optimizer = optim.Adam([
    {'params': cnn_model.parameters()},
    {'params': projection_layer_model.parameters()},
    {'params': bottom_up_mlps_models.parameters()},
    {'params': feedback_mlps_models.parameters()}
], lr=lr)

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training loop
# Set epochs to a small number (e.g., 1) for quick testing if needed
# epochs = 1 
for epoch in range(epochs):
    # Set models to training mode
    cnn_model.train()
    projection_layer_model.train()
    bottom_up_mlps_models.train()
    feedback_mlps_models.train()

    for batch_idx, (data, target_labels) in enumerate(train_loader):
        current_device = next(cnn_model.parameters()).device 
        data = data.to(current_device)
        target_labels = target_labels.to(current_device)

        x_features = cnn_model(data) 

        R_states = [torch.zeros(data.size(0), embed_dim, device=current_device) for _ in range(T)]
        R_states[0] = projection_layer_model(x_features.detach()).clone() 

        for _ in range(K):
            R_states_old = [r.clone().detach() for r in R_states] 

            P_bu_inf_0 = projection_layer_model(x_features.detach())
            error_bu_inf_0 = R_states_old[0] - P_bu_inf_0
            R_states[0] = R_states_old[0] - lr_inference_bu * error_bu_inf_0 

            for l_inf_bu in range(1, T):
                P_bu_inf_l = bottom_up_mlps_models[l_inf_bu](x_features.detach(), R_states_old[l_inf_bu-1])
                error_bu_inf_l = R_states_old[l_inf_bu] - P_bu_inf_l
                R_states[l_inf_bu] = R_states_old[l_inf_bu] - lr_inference_bu * error_bu_inf_l

            for l_inf_td in range(T - 1): 
                P_td_inf_l = feedback_mlps_models[l_inf_td](x_features.detach(), R_states_old[l_inf_td+1])
                error_td_inf_l = R_states_old[l_inf_td] - P_td_inf_l
                R_states[l_inf_td] = R_states_old[l_inf_td] - lr_inference_td * error_td_inf_l
        
        R_detached_for_loss = [r.clone().detach().requires_grad_(True) for r in R_states]

        u_y = torch.zeros(data.size(0), embed_dim, device=current_device).scatter_(1, target_labels.unsqueeze(1), 1)
        L_target = nn.MSELoss()(R_detached_for_loss[T-1], u_y)

        L_bu_predictive = torch.tensor(0.0, device=current_device)
        initial_R0_candidate_for_mlp0 = projection_layer_model(x_features) 
        P_bu_loss_0 = bottom_up_mlps_models[0](x_features, initial_R0_candidate_for_mlp0.detach())
        L_bu_predictive = L_bu_predictive + nn.MSELoss()(P_bu_loss_0, R_detached_for_loss[0].detach())

        for l_bu_loss in range(1, T):
            P_bu_loss_l = bottom_up_mlps_models[l_bu_loss](x_features, R_detached_for_loss[l_bu_loss-1].detach())
            L_bu_predictive = L_bu_predictive + nn.MSELoss()(P_bu_loss_l, R_detached_for_loss[l_bu_loss].detach())

        L_td_predictive = torch.tensor(0.0, device=current_device)
        for l_td_loss in range(T - 1): 
            P_td_loss_l = feedback_mlps_models[l_td_loss](x_features, R_detached_for_loss[l_td_loss+1].detach())
            L_td_predictive = L_td_predictive + nn.MSELoss()(P_td_loss_l, R_detached_for_loss[l_td_loss].detach())
        
        total_loss = L_target + lambda_bu_predictive * L_bu_predictive + lambda_td_predictive * L_td_predictive
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {total_loss.item():.4f}, L_target: {L_target.item():.4f}')

print("Training Finished")

# Prediction function
def predict(data_input, cnn, projection_layer, bottom_up_mlps, feedback_mlps):
    # Set models to evaluation mode
    cnn.eval()
    projection_layer.eval()
    bottom_up_mlps.eval()
    feedback_mlps.eval()

    current_device = next(cnn.parameters()).device

    # Ensure data_input is a batch of size 1 and on the correct device
    # Input data_input is expected to be a single image tensor, e.g., (C, H, W)
    if data_input.ndim == 3: 
        data_input_batched = data_input.unsqueeze(0).to(current_device)
    elif data_input.ndim == 4 and data_input.size(0) == 1: # Already (1, C, H, W)
        data_input_batched = data_input.to(current_device)
    else:
        raise ValueError(f"data_input has unexpected shape: {data_input.shape}. Expecting (C,H,W) or (1,C,H,W).")

    with torch.no_grad():
        x_features = cnn(data_input_batched)

        # Initialize R_states for inference (batch size is 1)
        R_states = [torch.zeros(1, embed_dim, device=current_device) for _ in range(T)]
        R_states[0] = projection_layer(x_features).clone() 

        # Iterative Inference Process (K steps)
        for _ in range(K):
            R_states_old = [r.clone() for r in R_states] 

            # Bottom-up updates
            P_bu_inf_0 = projection_layer(x_features) 
            error_bu_inf_0 = R_states_old[0] - P_bu_inf_0
            R_states[0] = R_states_old[0] - lr_inference_bu * error_bu_inf_0 

            for l_inf_bu in range(1, T):
                P_bu_inf_l = bottom_up_mlps[l_inf_bu](x_features, R_states_old[l_inf_bu-1])
                error_bu_inf_l = R_states_old[l_inf_bu] - P_bu_inf_l
                R_states[l_inf_bu] = R_states_old[l_inf_bu] - lr_inference_bu * error_bu_inf_l

            # Top-down updates
            for l_inf_td in range(T - 1): 
                P_td_inf_l = feedback_mlps[l_inf_td](x_features, R_states_old[l_inf_td+1])
                error_td_inf_l = R_states_old[l_inf_td] - P_td_inf_l
                R_states[l_inf_td] = R_states_old[l_inf_td] - lr_inference_td * error_td_inf_l
        
        prediction = torch.argmax(R_states[T-1], dim=1)
        return prediction.item()

# Demonstration of the predict function
if __name__ == '__main__':
    print("\n--- Running Prediction Demo ---")
    
    demo_device = torch.device("cpu") 
    if epochs > 0: # Check if training actually ran and models might be on GPU
        try:
            demo_device = next(cnn_model.parameters()).device
            print(f"Models are on device: {demo_device} (from training).")
        except StopIteration: 
            print("Models have no parameters. Forcing demo device to CPU and moving models.")
            cnn_model.to(demo_device)
            projection_layer_model.to(demo_device)
            bottom_up_mlps_models.to(demo_device)
            feedback_mlps_models.to(demo_device)
    else: # epochs == 0, so no training, ensure models are on CPU for demo
        print(f"Epochs is 0. Ensuring models are on device: {demo_device} for prediction demo.")
        cnn_model.to(demo_device)
        projection_layer_model.to(demo_device)
        bottom_up_mlps_models.to(demo_device)
        feedback_mlps_models.to(demo_device)

    if not train_loader: 
        print("Train loader is not available. Skipping prediction demo with DataLoader sample.")
    else:
        try:
            data_iterator = iter(train_loader)
            sample_images, sample_labels = next(data_iterator)
            
            single_image_data = sample_images[0].to(demo_device) 
            true_label = sample_labels[0].item()

            print(f"Using a sample image with true label: {true_label}")

            predicted_label = predict(single_image_data, 
                                      cnn_model, 
                                      projection_layer_model, 
                                      bottom_up_mlps_models, 
                                      feedback_mlps_models)
            
            print(f"Predicted label: {predicted_label}")

            if predicted_label == true_label:
                print("Prediction was CORRECT!")
            else:
                print("Prediction was INCORRECT.")

        except Exception as e:
            print(f"Could not run prediction demo with DataLoader sample: {e}")
    
    print("\n--- Prediction Demo with Dummy Tensor ---")
    try:
        # Ensure models are on the demo_device for this dummy test as well
        cnn_model.to(demo_device) 
        projection_layer_model.to(demo_device)
        bottom_up_mlps_models.to(demo_device)
        feedback_mlps_models.to(demo_device)

        # Create a dummy image tensor with the expected shape (C, H, W) for MNIST
        dummy_image_tensor = torch.randn(1, 28, 28).to(demo_device) 
        print(f"Using a dummy tensor of shape {dummy_image_tensor.shape} on device {dummy_image_tensor.device} for prediction.")
        
        predicted_label_dummy = predict(dummy_image_tensor, 
                                        cnn_model, 
                                        projection_layer_model, 
                                        bottom_up_mlps_models, 
                                        feedback_mlps_models)
        print(f"Predicted label for dummy tensor: {predicted_label_dummy}")
    except Exception as e_dummy:
        print(f"Error during dummy tensor prediction: {e_dummy}")

    print("--- Prediction Demo Finished ---")
