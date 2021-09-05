from scripts.definitions.models.m_classic import MClassic
from scripts.definitions.models.m_classic_drop1 import MClassicDrop1
from scripts.definitions.models.m_classic_drop_old import MClassicDropOld
from scripts.definitions.models.m_classic_less_layers import MClassicLessLayers
from scripts.definitions.models.m_classic_less_layers_drop import MClassicLessLayersDrop
from scripts.definitions.models.m_classic_less_nodes import MClassicLessNodes
from scripts.definitions.models.m_classic_small_both_drop import MClassicSmallBothDrop
from scripts.definitions.models.m_classic_small_class_drop import MClassicSmallClassDrop
from scripts.definitions.models.m_classic_small_feature_drop import MClassicSmallFeatureDrop
from scripts.definitions.models.m_classic_xavier import MClassicXavier


def create_model(model_name, input_size):
    if model_name == "m_classic":
        return MClassic.create(input_size).to('cuda')

    if model_name == "m_classic_drop1":
        return MClassicDrop1.create(input_size).to('cuda')

    if model_name == "m_classic_drop_old":
        return MClassicDropOld.create(input_size).to('cuda')

    if model_name == "m_classic_less_layers":
        return MClassicLessLayers.create(input_size).to('cuda')

    if model_name == "m_classic_less_nodes":
        return MClassicLessNodes.create(input_size).to('cuda')

    if model_name == "m_classic_xavier":
        return MClassicXavier.create(input_size).to('cuda')

    if model_name == "m_classic_less_layers_drop":
        return MClassicLessLayersDrop.create(input_size).to('cuda')

    if model_name == "m_classic_small_both_drop":
        return MClassicSmallBothDrop.create(input_size).to('cuda')

    if model_name == "m_classic_small_class_drop":
        return MClassicSmallClassDrop.create(input_size).to('cuda')

    if model_name == "m_classic_small_feature_drop":
        return MClassicSmallFeatureDrop.create(input_size).to('cuda')