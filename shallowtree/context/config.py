from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import yaml

if TYPE_CHECKING:
    from shallowtree.utils.type_utils import Any, StrDict


@dataclass
class Configuration:
    """
    Encapsulating the settings of the tree search, including the policy,
    the stock, the loaded scorers and various parameters.
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Configuration):
            return False
        for key, setting in vars(self).items():
            if isinstance(setting, (int, float, str, bool, list)):
                if (
                    vars(self)[key] != vars(other)[key]
                ):
                    return False
        return True

    @classmethod
    def from_file(cls, filename: str) -> StrDict:
        """
        Loads a configuration from a yaml file.
        The parameters not set in the yaml file are taken from the default values.
        The policies and stocks specified in the yaml file are directly loaded.
        The parameters in the yaml file may also contain environment variables as
        values.

        :param filename: the path to a yaml file
        :return: a Configuration object with settings from the yaml file
        :raises:
            ValueError: if parameter's value expects an environment variable that
                does not exist in the current environment
        """
        with open(filename, "r") as fileobj:
            txt = fileobj.read()
        print(filename+ 80*"#")
        environ_var = re.findall(r"\$\{.+?\}", txt)
        for item in environ_var:
            if item[2:-1] not in os.environ:
                raise ValueError(f"'{item[2:-1]}' not in environment variables")
            txt = txt.replace(item, os.environ[item[2:-1]])
        _config = yaml.load(txt, Loader=yaml.SafeLoader)
        return _config

    @classmethod
    def from_json(self, path: str) -> Dict:
        with open(path) as f:
            json_input = f.read().replace('\r', '').replace('\n', '')
        try:
            return json.loads(json_input)
        except (ValueError, KeyError, TypeError) as e:
            print(f"JSON format error in file ${path}: \n ${e}")