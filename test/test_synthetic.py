import pytest
import numpy as np
from loguru import logger


@pytest.fixture(scope="class")
def sthp():
    from data.synthetic import STHPDataset
    _sthp = STHPDataset(s_mu=np.array([0, 0]), 
                        g0_cov=np.array([[5, 0],
                                         [0, 5]]),
                        g2_cov=np.array([[.1, 0],
                                         [0, .1]]),
                        alpha=.5, beta=.6, mu=.15,
                        dist_only=False)
    _sthp.load('data/raw/spatiotemporal/sthp1.data', t_start=0, t_end=10000)
    return _sthp


class TestClass:
    
    @staticmethod
    def test_g2(sthp):
        s_1 = np.array([0.1, 0.1])
        his_s_1 = np.array([[0.1, 0.1], [-0.1, -0.1]])
        expect_1 = np.array([0.03183099, 0.03157736])
        actual_1 = sthp.g2(s_1, his_s_1, sthp.g0_sidc, sthp.g0_ic)
        assert expect_1.shape == actual_1.shape
        assert np.allclose(actual_1, expect_1)
        
        s_2 = np.array([0.0, 0.0])
        his_s_2 = his_s_1
        expect_2 = np.array([0.03176739, 0.03176739])
        actual_2 = sthp.g2(s_2, his_s_2, sthp.g0_sidc, sthp.g0_ic)
        assert expect_2.shape == actual_2.shape
        assert np.allclose(actual_2, expect_2)

        # Batch test 1 (N s and N his_s)
        s_3 = np.stack([s_1, s_2], 0)
        his_s_3 = np.stack([his_s_1, his_s_2], 0)
        expect_3 = np.stack([expect_1, expect_2], 0)
        actual_3 = sthp.g2(s_3, his_s_3, sthp.g0_sidc, sthp.g0_ic)
        assert expect_3.shape == actual_3.shape
        assert np.all(np.isclose(actual_3, expect_3))
        
        # Batch test 2 (N s and one his_s)
        s_4 = s_3
        his_s_4 = his_s_1
        expect_4 = expect_3
        actual_4 = sthp.g2(s_4, his_s_4, sthp.g0_sidc, sthp.g0_ic)
        assert expect_4.shape == actual_4.shape
        assert np.all(np.isclose(actual_4, expect_4))
        
        # Batch test 3 (one s and N his_s)
        s_5 = s_1
        his_s_5 = np.stack([his_s_1, his_s_1], 0)
        expect_5 = np.stack([expect_1, expect_1], 0)
        actual_5 = sthp.g2(s_5, his_s_5, sthp.g0_sidc, sthp.g0_ic)
        assert expect_5.shape == actual_5.shape
        assert np.all(np.isclose(actual_5, expect_5))
        
    @staticmethod
    def test_g0(sthp):
        s_1 = np.array([0.1, 0.1])
        expect_1 = np.array([0.03176739])
        actual_1 = sthp.g0(s_1, sthp.s_mu, sthp.g0_sidc, sthp.g0_ic)
        assert expect_1.shape == actual_1.shape
        assert np.allclose(actual_1, expect_1)
        
        # Batch test
        s_2 = np.array([[0.1, 0.1], [0.0, 0.0]])
        expect_2 = np.array([0.03176739, 0.03183099])
        actual_2 = sthp.g0(s_2, sthp.s_mu, sthp.g0_sidc, sthp.g0_ic)
        assert expect_2.shape == actual_2.shape
        assert np.allclose(actual_2, expect_2)
        
    @staticmethod
    def test_g1(sthp):
        t_1 = 10.0
        his_t_1 = np.array([2.0, 3.0])
        expect_1 = np.array([0.00411487, 0.00749779])
        actual_1 = sthp.g1(t_1, his_t_1, sthp.alpha, sthp.beta)
        assert expect_1.shape == actual_1.shape
        assert np.allclose(actual_1, expect_1)
        
        t_2 = 5.0
        his_t_2 = np.array([2.0, 3.0])
        expect_2 = np.array([0.08264944, 0.15059711])
        actual_2 = sthp.g1(t_2, his_t_2, sthp.alpha, sthp.beta)
        assert expect_2.shape == actual_2.shape
        assert np.allclose(actual_2, expect_2)

        # Batch test 1 (N t and N his_t)
        t_3 = np.array([t_1, t_2])
        his_t_3 = np.stack([his_t_1, his_t_2], 0)
        expect_3 = np.stack([expect_1, expect_2], 0)
        actual_3 = sthp.g1(t_3, his_t_3, sthp.alpha, sthp.beta)
        assert expect_3.shape == actual_3.shape
        assert np.allclose(actual_3, expect_3)
        
        # Batch test 2 (N t and one his_t)
        t_4 = np.array([t_1, t_2])
        his_t_4 = his_t_1
        expect_4 = np.stack([expect_1, expect_2], 0)
        actual_4 = sthp.g1(t_4, his_t_4, sthp.alpha, sthp.beta)
        assert expect_4.shape == actual_4.shape
        assert np.allclose(actual_4, expect_4)

        # Batch test 3 (one t and N his_t)
        t_5 = t_1
        his_t_5 = np.stack([his_t_1, his_t_1], 0)
        expect_5 = np.stack([expect_1, expect_1], 0)
        actual_5 = sthp.g1(t_5, his_t_5, sthp.alpha, sthp.beta)
        assert expect_5.shape == actual_5.shape
        assert np.allclose(actual_5, expect_5)

    @staticmethod
    def test_lamb_st(sthp):
        for t_1 in [10.0, np.array([10.0]), np.array(10.0)]:
            s_1 = np.array([0.0, 0.0])
            expect_1 = np.array([0.00479825])
            actual_1 = sthp.lamb_st(s_1, t_1)
            assert expect_1.shape == actual_1.shape
            assert np.allclose(actual_1, expect_1)
        
        t_2 = 15.0
        s_2 = np.array([0.0, 1.0])
        expect_2 = np.array([0.00765035])
        actual_2 = sthp.lamb_st(s_2, t_2)
        assert expect_2.shape == actual_2.shape
        assert np.allclose(actual_2, expect_2)
        
        # Batch test 1 (one t and N s)
        t_3 = t_1
        s_3 = np.stack([s_1, s_2], 0)
        expect_3 = np.stack([expect_1.item(), 0.05426189], 0)
        actual_3 = sthp.lamb_st(s_3, t_3)
        assert expect_3.shape == actual_3.shape
        assert np.allclose(actual_3, expect_3)
        
        # Batch test 2 (N t and one s)
        t_4 = np.array([t_1, t_2])
        s_4 = s_2
        expect_4 = np.stack([expect_3[1].item(), expect_2.item()], 0)
        actual_4 = sthp.lamb_st(s_4, t_4)
        assert expect_4.shape == actual_4.shape
        assert np.allclose(actual_4, expect_4)
        
        # Batch test 3 (N t and N s)
        t_5 = np.array([t_1, t_2])
        s_5 = np.stack([s_1, s_2], 0)
        expect_5 = np.stack([expect_1.item(), expect_2.item()], 0)
        actual_5 = sthp.lamb_st(s_5, t_5)
        assert expect_5.shape == actual_5.shape
        assert np.allclose(actual_5, expect_5)
