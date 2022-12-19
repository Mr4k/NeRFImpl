import unittest
import torch

from neural_nerf import embed_tensor


class TestNeuralNerfUnit(unittest.TestCase):
    def test_embed_tensor(self):
        epsilon = 0.00001
        p = torch.tensor([[0.5, 0.1, 0.75], [0.8, 0.4, 0.6]])
        expected = torch.stack(
            [
                torch.concat(
                    [
                        torch.sin(
                            torch.pi
                            * torch.tensor(
                                [
                                    p[0, 0],
                                    p[0, 1],
                                    p[0, 2],
                                    2.0 * p[0, 0],
                                    2.0 * p[0, 1],
                                    2.0 * p[0, 2],
                                    4.0 * p[0, 0],
                                    4.0 * p[0, 1],
                                    4.0 * p[0, 2],
                                    8.0 * p[0, 0],
                                    8.0 * p[0, 1],
                                    8.0 * p[0, 2],
                                ],
                                dtype=torch.float,
                            )
                        ),
                        torch.cos(
                            torch.pi
                            * torch.tensor(
                                [
                                    p[0, 0],
                                    p[0, 1],
                                    p[0, 2],
                                    2.0 * p[0, 0],
                                    2.0 * p[0, 1],
                                    2.0 * p[0, 2],
                                    4.0 * p[0, 0],
                                    4.0 * p[0, 1],
                                    4.0 * p[0, 2],
                                    8.0 * p[0, 0],
                                    8.0 * p[0, 1],
                                    8.0 * p[0, 2],
                                ],
                                dtype=torch.float,
                            )
                        ),
                    ]
                ),
                torch.concat(
                    [
                        torch.sin(
                            torch.pi
                            * torch.tensor(
                                [
                                    p[1, 0],
                                    p[1, 1],
                                    p[1, 2],
                                    2.0 * p[1, 0],
                                    2.0 * p[1, 1],
                                    2.0 * p[1, 2],
                                    4.0 * p[1, 0],
                                    4.0 * p[1, 1],
                                    4.0 * p[1, 2],
                                    8.0 * p[1, 0],
                                    8.0 * p[1, 1],
                                    8.0 * p[1, 2],
                                ],
                                dtype=torch.float,
                            )
                        ),
                        torch.cos(
                            torch.pi
                            * torch.tensor(
                                [
                                    p[1, 0],
                                    p[1, 1],
                                    p[1, 2],
                                    2.0 * p[1, 0],
                                    2.0 * p[1, 1],
                                    2.0 * p[1, 2],
                                    4.0 * p[1, 0],
                                    4.0 * p[1, 1],
                                    4.0 * p[1, 2],
                                    8.0 * p[1, 0],
                                    8.0 * p[1, 1],
                                    8.0 * p[1, 2],
                                ],
                                dtype=torch.float,
                            )
                        ),
                    ]
                ),
            ]
        )

        result = embed_tensor(p, 4)
        self.assertEqual(result.shape, torch.Size([2, 24]))
        self.assertLess((result - expected).abs().sum(), epsilon)


if __name__ == "__main__":
    unittest.main()
