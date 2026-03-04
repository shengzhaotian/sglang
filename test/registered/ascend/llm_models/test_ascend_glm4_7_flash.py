import os
import sys

# Add local repository to PYTHONPATH for running on current repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))
os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "../../../../python") + ":" + os.environ.get("PYTHONPATH", "")

import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

# GLM-4.7-Flash model weights path
GLM_4_7_FLASH_WEIGHTS_PATH = "/home/trae/testCode/weight/GLM4-7-flash"

register_npu_ci(
    est_time=600,
    suite="nightly-1-npu-a3",
    nightly=True,
)


class TestGLM47Flash(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the GLM-4.7-Flash model on the GSM8K dataset.

    [Test Category] Model
    [Test Target] GLM-4.7-Flash (Glm4MoeLiteForCausalLM)
    [Hardware] NPU (Ascend)
    """

    model = GLM_4_7_FLASH_WEIGHTS_PATH
    accuracy = 0.65  # Expected accuracy threshold for MoE model
    timeout_for_server_launch = 600  # Longer timeout for larger model
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",  # Higher memory fraction needed for MoE model
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]
    gsm8k_num_shots = 5


if __name__ == "__main__":
    unittest.main()
