import torch.nn as nn
import timm
    
class PokemonClassifier(nn.Module):
    def __init__(self, num_classes=149):
        super(PokemonClassifier, self).__init__()
        # Define base model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        eneT_out_size = 1280

        # Make classifier
        self.classifier = nn.Linear(eneT_out_size, num_classes)
        

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output