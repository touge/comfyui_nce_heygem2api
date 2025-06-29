from .nodes import *

NODE_CONFIG = {
    "NCEHeygemConfigure": {
        "class": NCEHeygemConfigure,
        "name": "接口配置",  
    },
    "NCEHeygemGenerateVideo": {
        "class": NCEHeygemGenerateVideo,
        "name": "视频生成",
    },
    # "NCEHeygem2apiGenSpeaker": {
    #     "class": NCEHeygem2apiGenSpeaker,
    #     "name": "上传模特",
    # },
    # "NCEHeygem2apiSpeakers": {
    #     "class": NCEHeygem2apiSpeakers,
    #     "name": "模特列表",
    # },  
}

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)
WEB_DIRECTORY = "./js"


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("🐍 Heygem Generate Video Api Node Initialized")




