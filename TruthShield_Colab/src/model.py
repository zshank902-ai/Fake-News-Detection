import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import XLMRobertaModel
import torchvision.models as models
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class AdversarialFakeNewsModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_classes=1, num_sources=101):
        super(AdversarialFakeNewsModel, self).__init__()
        # Load Base XLM-R
        base_model = XLMRobertaModel.from_pretrained(model_name)
        
        # Apply LoRA (Intelligence King)
        peft_config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=["query", "value"], 
            lora_dropout=0.05, 
            bias="none"
        )
        self.xlm_roberta = get_peft_model(base_model, peft_config)
        
        # --- Metadata Branch (Source/Author) ---
        self.source_embedding = nn.Embedding(num_sources, 32)
        
        # --- Vision Branch ---
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.vision_projector = nn.Linear(512, 128)
        
        # --- Audio & Video Fusion (Phase 3 Readiness) ---
        self.audio_projector = nn.Linear(1024, 128) # For Wav2Vec2/Hubert
        self.video_projector = nn.Linear(2048, 128) # For 3D-CNN/Video-Transformer
        
        # --- Knowledge Buffer Stream (Phase 3: Contextual Retrieval) ---
        self.knowledge_projector = nn.Linear(768, 128) # For Verified News Embeddings
        
        # --- Fusion & Attention ---
        self.self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        # --- Gated Fusion Intelligence (The Modality Judge) ---
        self.gating_network = nn.Sequential(
            nn.Linear(1312, 5), # 5 streams: Text, Source, Image, Audio, Video+Knowledge
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1312, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # --- Efficiency: Static Feature Buffers ---
        self.register_buffer("empty_image", torch.zeros(1, 128))
        self.register_buffer("empty_audio", torch.zeros(1, 128))
        self.register_buffer("empty_video", torch.zeros(1, 128))
        self.register_buffer("empty_knowledge", torch.zeros(1, 128))
        self.register_buffer("empty_source", torch.zeros(1, 32))
        
        # --- Adversarial Branch (Language Discriminator) ---
        self.lang_discriminator = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, images=None, audio_feats=None, video_feats=None, knowledge_feats=None, source_ids=None, alpha=1.0, mc_dropout=False):
        # 1. Text Features
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply Self-Attention
        attn_output, _ = self.self_attention(sequence_output, sequence_output, sequence_output)
        pooled_output = attn_output[:, 0, :]
        
        # 2. Vision Features (Smart Selection)
        if images is not None:
            vis_features = self.resnet(images)
            image_features = self.vision_projector(vis_features)
        else:
            image_features = self.empty_image.expand(pooled_output.size(0), -1)
            
        # 3. Audio, Video & Knowledge (Zero-Latency Buffering)
        audio_features = self.audio_projector(audio_feats) if audio_feats is not None else self.empty_audio.expand(pooled_output.size(0), -1)
        video_features = self.video_projector(video_feats) if video_feats is not None else self.empty_video.expand(pooled_output.size(0), -1)
        knowledge_features = self.knowledge_projector(knowledge_feats) if knowledge_feats is not None else self.empty_knowledge.expand(pooled_output.size(0), -1)
            
        # 4. Source Metadata
        source_feats = self.source_embedding(source_ids) if source_ids is not None else self.empty_source.expand(pooled_output.size(0), -1)
            
        # 5. Gated Multimodal Fusion (The Magic Happens Here)
        # We first concatenate, then let the Gating Network decide weights
        raw_combined = torch.cat((pooled_output, source_feats, image_features, audio_features, video_features, knowledge_features), dim=1)
        
        gate_weights = self.gating_network(raw_combined) # [batch, 5]
        
        # Apply weights to modalities (simplified version for speed)
        # In this advanced version, the gating network adjusts the influence of each stream
        combined_features = raw_combined * gate_weights.repeat_interleave(torch.tensor([768, 32, 128, 128, 256]).to(pooled_output.device), dim=1)
        
        # 6. Output
        label_output = self.classifier(combined_features)
        
        # 6. Adversarial Branch
        reverse_features = grad_reverse(pooled_output, alpha)
        lang_logits = self.lang_discriminator(reverse_features)
        
        contrastive_features = pooled_output
        return label_output, lang_logits, contrastive_features

class BaselineFakeNewsModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_classes=1):
        super(BaselineFakeNewsModel, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_name)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cnn_input = sequence_output.transpose(1, 2)
        cnn_output = self.cnn(cnn_input)
        lstm_input = cnn_output.transpose(1, 2)
        _, (hn, _) = self.lstm(lstm_input)
        logits = self.fc(hn.squeeze(0))
        return logits

    def freeze_backbone(self, num_layers=6):
        for param in self.xlm_roberta.embeddings.parameters():
            param.requires_grad = False
        for i in range(num_layers):
            for param in self.xlm_roberta.encoder.layer[i].parameters():
                param.requires_grad = False
