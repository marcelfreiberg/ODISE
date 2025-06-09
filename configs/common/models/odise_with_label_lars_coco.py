# LARS COCO specific model configuration with correct num_classes
from detectron2.config import LazyCall as L
from odise.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor
from odise.modeling.backbone.feature_extractor import FeatureExtractorBackbone
from .mask_generator_with_label import model
from odise.data.build import get_openseg_labels
from detectron2.data import MetadataCatalog

# Import the base model and override num_classes for LARS COCO (11 classes)
model.sem_seg_head.num_classes = 11

model.backbone = L(FeatureExtractorBackbone)(
    feature_extractor=L(LdmImplicitCaptionerExtractor)(
        encoder_block_indices=(5, 7),
        unet_block_indices=(2, 5, 8, 11),
        decoder_block_indices=(2, 5),
        steps=(0,),
        learnable_time_embed=True,
        num_timesteps=1,
        clip_model_name="ViT-L-14-336",
    ),
    out_features=["s2", "s3", "s4", "s5"],
    use_checkpoint=True,
    slide_training=True,
)
model.sem_seg_head.pixel_decoder.transformer_in_features = ["s3", "s4", "s5"]
model.clip_head.alpha = 0.3
model.clip_head.beta = 0.7

model.category_head.labels = L(get_openseg_labels)(dataset="lars_coco", prompt_engineered=False)
model.metadata = L(MetadataCatalog.get)(name="lars_coco_train_panoptic_with_sem_seg") 