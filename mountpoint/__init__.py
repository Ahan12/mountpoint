"""mountpoint -- training-free articulation estimation for functional elements.

    from mountpoint import predict_motion
    p = predict_motion('hook_pull', element_points, neighbourhood_points)
    p['motion_type']    'rot' | 'trans'
    p['motion_dir']     unit axis
    p['motion_origin']  a point on the axis, or None for translations
"""
from .motion import predict_motion, predict_motion_given_panel, signed_dir, OE, MD

__all__ = ['predict_motion', 'predict_motion_given_panel', 'signed_dir', 'OE', 'MD']
__version__ = '0.1.0'
