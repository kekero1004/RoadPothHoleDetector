# -*- coding: utf-8 -*-
"""
Road Pothole Detector - QGIS Plugin
AI-based Road Pothole/Crack Detection & Maintenance Route Optimization

This plugin provides automatic detection of road defects and optimal
maintenance route generation for road maintenance management.
"""


def classFactory(iface):
    """Load RoadPotholeDetector class from main module.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    :returns: RoadPotholeDetector plugin instance.
    :rtype: RoadPotholeDetector
    """
    from .main import RoadPotholeDetector
    return RoadPotholeDetector(iface)
