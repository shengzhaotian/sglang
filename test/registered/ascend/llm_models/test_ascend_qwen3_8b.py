import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

QWEN3_8B_WEIGHTS_PATH = "/root/.cache/modelscope/Qwen/Qwen3-8B"


class TestQwen38B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-8B model on the GSM8K dataset is no less than 0.75.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-8B
    """

    model = QWEN3_8B_WEIGHTS_PATH
    accuracy = 0.75
    other_args = [
        "--chunked-prefill-size",
        256,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
