from timm.models import EfficientNet
from timm.models.efficientnet import efficientnet_b0, efficientnet_b7, mixnet_m, mnasnet_a1
from timm.models.mobilenetv3 import mobilenetv3_large_100
from timm.models.efficientnet_builder import decode_arch_def
from timm.models.efficientnet_blocks import resolve_bn_args


def get_timm_models(name, dropout, drop_connect, bn_momentum):
	EfficientNet_b0 = efficientnet_b0(drop_rate=dropout, drop_connect_rate=drop_connect, bn_momentum=bn_momentum)
	EfficientNet_b7 = efficientnet_b0(drop_rate=dropout, drop_connect_rate=drop_connect, bn_momentum=bn_momentum)
	MixNet_m = mixnet_m(drop_rate=dropout, drop_connect_rate=drop_connect, bn_momentum=bn_momentum)
	MobileNet_v3 = mobilenetv3_large_100(drop_rate=dropout, drop_connect_rate=drop_connect, bn_momentum=bn_momentum)
	MNASNet_a1 = mnasnet_a1(drop_rate=dropout, drop_connect_rate=drop_connect, bn_momentum=bn_momentum)
	
	model_dict = {
		'efficientnet_b0': EfficientNet_b0,
		'efficientnet_b7': EfficientNet_b7,
		'mixnet_m': MixNet_m,
		'mobilenet_v3': MobileNet_v3,
		'mnasnet_a1': MNASNet_a1,
	}
	
	return model_dict[name]