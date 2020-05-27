from timm.models import EfficientNet
from timm.models.efficientnet import efficientnet_b0
from timm.models.efficientnet_builder import decode_arch_def
from timm.models.efficientnet_blocks import resolve_bn_args


def get_timm_models(name, dropout, drop_connect, bn_momentum):
	EfficientNet_b0 = efficientnet_b0(drop_rate=dropout, drop_connect_rate=drop_connect,
	                                  bn_momentum=bn_momentum)
	
	model_dict = {
		'efficientnet_b0': EfficientNet_b0
	}
	
	return model_dict[name]