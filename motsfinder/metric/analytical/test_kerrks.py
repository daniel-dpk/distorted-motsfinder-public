#!/usr/bin/env python3 

import unittest 
import sys 

import numpy as np 

from testutils import DpkTestCase, slowtest 
from .kerrks import KerrKSSlice 
from motsfinder.utils import printoptions

class TestKerrks(DpkTestCase): 
    
    def test_kerrksslice_curv_nonSpecialPnt(self): 
        g = KerrKSSlice(M = 1.0, a = 0.5) 
        curv = g.get_curv() 
        p = [0.23, 0.0, 0.34] 
        result = curv(p) 
        expected = np.array([[ 1.476929378132757, -0.14218477046658,   0.500706282710333],
                             [-0.14218477046658,   1.206279098325951,  0.044364471323255],
                             [ 0.500706282710333,  0.044364471323255,  1.362414985985481]]     
        ) 
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14) 
    
    def test_kerrksslice_curv_SpecialPnt1(self):
        g = KerrKSSlice(M = 1.0, a = 0.5) 
        curv = g.get_curv() 
        p = [0.0, 0.0, 1.5] 
        result = curv(p) 
        expected = np.array([[ 0.485423900973534, -0.               , -0.               ],
                             [-0.               ,  0.485423900973534, -0.               ],
                             [-0.               , -0.               , -0.69038065916236 ]]     
        ) 
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14) 
    
    def test_kerrksslice_curv_SpecialPnt2(self):
        g = KerrKSSlice(M = 1.0, a = 0.5) 
        curv = g.get_curv() 
        p = [1.3, 0.0, 0.0] 
        result = curv(p) 
        expected = np.array([[-1.328618696775885,  0.730782221948226, -0.               ],
                             [ 0.730782221948226,  0.619854303609456, -0.               ],
                             [-0.               , -0.               ,  0.850517271799714]]  
        ) 
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14) 
       
        
    def test_kerrksslice_lapse_nonSpecialPnt(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        lapse = g.get_lapse()
        p = [0.14, 0.0, 0.87]
        result = lapse(p)
        expected = np.array([0.6054802933332376])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_lapse_SpecialPnt1(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        lapse = g.get_lapse()
        p = [0.0, 0.0, 1.5]
        result = lapse(p)

        expected = np.array([0.674199862463242])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_lapse_SpecialPnt2(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        lapse = g.get_lapse()
        p = [1.3, 0.0, 0.0]
        result = lapse(p)
        expected = np.array([0.6123724356957945])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_shift_nonSpecialPnt(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        shift = g.get_shift()
        p = [0.14, 0.0, 0.87]
        result = shift(p)
        expected = np.array([ 0.07624335265557,  -0.043395689029844,  0.627288797997605])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_shift_SpecialPnt1(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        shift = g.get_shift()
        p = [0.0, 0.0, 1.5]
        result = shift(p)
        
        expected = np.array([0.               , 0.               , 0.545454545454545])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
        
    def test_kerrksslice_shift_SpecialPnt2(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        shift = g.get_shift()
        p = [1.3, 0.0, 0.0]
        result = shift(p)
        
        expected = np.array([ 0.576923076923077, -0.240384615384615,  0.               ])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14) 

    def test_kerrksslice_metric_nonSpecialPnt(self):
        p = [0.14, 0.0, 0.87]
        g = KerrKSSlice(M = 1.0, a = 0.5)
        result = g.at(point = p).mat
        expected = np.array([[ 1.025034000965719, -0.01424868769857,   0.205966131182575],
                             [-0.01424868769857,   1.00810997416711,  -0.117230445254118],
                             [ 0.205966131182575, -0.117230445254118,  2.694577197324969]]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_metric_SpecialPnt1(self):
        p = [0.0, 0.0, 1.5]
        g = KerrKSSlice(M = 1.0, a = 0.5)
        result = g.at(point = p).mat
        expected = np.array([[1. , 0. , 0. ],
                             [0. , 1. , 0. ],
                             [0. , 0. , 2.2]]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_metric_SpecialPnt2(self):
        p = [1.3, 0.0, 0.0]
        g = KerrKSSlice(M = 1.0, a = 0.5)
        result = g.at(point = p).mat
        expected = np.array([[ 2.420118343195266, -0.591715976331361,  0.               ],
                             [-0.591715976331361 , 1.2465483234714   , 0.               ],
                             [ 0.                , 0.                , 1.               ]]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_dtlapse_nonSpecialPnt(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        dtlapse = g.get_dtlapse()
        p = [0.14, 0.0, 0.87]
        result = dtlapse(p)
        expected = np.array([0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_dtlapse_SpecialPnt1(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        dtlapse = g.get_dtlapse()
        p = [0.0, 0.0, 1.5]
        result = dtlapse(p)
        expected = np.array([0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_dtlapse_SpecialPnt2(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        dtlapse = g.get_dtlapse()
        p = [1.3, 0.0, 0.0]
        result = dtlapse(p)
        expected = np.array([0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
    
    def test_kerrksslice_dtshift_nonSpecialPnt(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        dtshift = g.get_dtshift()
        p = [0.14, 0.0, 0.87]
        result = dtshift(p)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_dtshift_SpecialPnt1(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        dtshift = g.get_dtshift()
        p = [0.0, 0.0, 1.5]
        result = dtshift(p)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_kerrksslice_dtshift_SpecialPnt2(self):
        g = KerrKSSlice(M = 1.0, a = 0.5)
        dtshift = g.get_dtshift()
        p = [1.3, 0.0, 0.0]
        result = dtshift(p)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    

def run_tests(): 
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]) 
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures) 


if __name__ == '__main__': 
    unittest.main() 
