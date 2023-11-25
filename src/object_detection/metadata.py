#!/usr/bin/env python3
"""
Code for gathering the metadata of a model.

Note that get_model_metadata should be called before each inference as the model server can dynamically update and
change the current version in use.
"""
# Standard Libraries
from os import environ
from os.path import dirname, join, pardir, realpath
from typing import NamedTuple, Tuple, Union

# Installed Libraries
from ovmsclient import make_grpc_client
from ovmsclient.tfs_compat.base.errors import ModelNotFoundError
from yaml import safe_load

# Local Files

# Create connection to the model server
HOST = environ.get("INFERENCE_HOST", "localhost")
PORT = environ.get("INFERENCE_PORTT", "9000")
CLIENT = make_grpc_client(f"{HOST}:{PORT}")

THIS_DIR = dirname(realpath(__file__))
MODELS_DIR = environ.get("MODELS_DIR", join(THIS_DIR, pardir, pardir, "machine_learning", "models"))

class ModelMetadata(NamedTuple):
    """
    Class for sharing model metadata.
    """
    name: str
    version: int
    input_name: str
    input_shape: Tuple[float]
    classes: Tuple[str]

def get_model_metadata(model_name: str) -> Union[ModelMetadata, None]:
    """
    Get the models metadata and return a ModelMetadata namedtuple.
    """
    try:
        metadata = CLIENT.get_model_metadata(model_name)

        with open(join(MODELS_DIR, model_name, str(metadata["model_version"]), "metadata.yaml"), "r") as input_yaml:
            metadata_yaml = safe_load(input_yaml)

        model_metadata = ModelMetadata(
            model_name,
            metadata["model_version"],
            next(iter(metadata["inputs"])),
            tuple(metadata_yaml["imgsz"]),
            tuple(metadata_yaml["names"][i] for i in range(len(metadata_yaml["names"])))
        )
        return model_metadata
    except ModelNotFoundError:
        print(f"Model '{model_name}' not found")
    except FileNotFoundError as exc:
        print(f"Cannot find metadata.yaml for '{model_name}', {str(exc)}")
    return None

def get_model_status(model_meta: Union[str, ModelMetadata]) -> Union[str, None]:
    """
    Get the models status.
    """
    try:
        if isinstance(model_meta, str):
            model_meta = get_model_metadata(model_meta)

        if model_meta is None:
            # No model metadata found, so no status either.
            return None

        status = CLIENT.get_model_status(model_meta.name)
        status = status[model_meta.version]["state"]
        return status
    except ModelNotFoundError:
        print(f"Model '{model_meta.name}' not found")
    return None

if __name__ == "__main__":
    print(get_model_metadata("handguns"))
    print(get_model_metadata("machineguns"))

    print(get_model_status("handguns"))
    print(get_model_status("machineguns"))
